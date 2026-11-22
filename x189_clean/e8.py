# SPDX-License-Identifier: Apache-2.0
"""Bounded tensor streaming for the eight-dimensional lattice codec."""
import hashlib
import struct

import numpy as np

from .lattice import nearest, prototypes, table_for


def sylvester(size):
    if size < 8 or size & (size - 1):
        raise ValueError("This record version accepts only power-of-two axes of at least eight")
    index = np.arange(size, dtype=np.uint32)
    parity = np.array([k.bit_count() % 2 for k in range(size)])
    return ((1 - 2 * parity[index[:, None] & index]) / np.sqrt(size)).astype(np.float32)


ROWS = 128
COLS = 512
HEAD = struct.Struct("<4sIIHH")
MAGIC = b"E8Q2"
RIDGE = .02
PANEL_BYTES = 2 + ROWS * COLS // 4


def geometry(rows, cols):
    if rows < ROWS or cols < COLS or rows % ROWS or cols % COLS:
        raise ValueError(f"Unaligned XE82 real shape {(rows, cols)}")
    return rows // ROWS, cols // COLS


def size(rows, cols):
    nr, nc = geometry(rows, cols)
    return HEAD.size + (rows + cols) // 8 + nr * nc * PANEL_BYTES


def prefix(su, sv):
    rows, cols = len(sv), len(su)
    geometry(rows, cols)
    if not np.isin(su, [-1, 1]).all() or not np.isin(sv, [-1, 1]).all():
        raise ValueError("Invalid XE82 signs")
    return (HEAD.pack(MAGIC, rows, cols, ROWS, COLS)
            + np.packbits(su < 0, bitorder="little").tobytes()
            + np.packbits(sv < 0, bitorder="little").tobytes())


def take_panels(raw, first, stop, ncols):
    # Interleaved 2-byte scale and u16 code array, one record per tile.
    dtype = np.dtype([("scale", "<f2"), ("ids", "<u2", (ROWS, COLS // 8))])
    packs = np.frombuffer(raw, dtype=dtype, count=(stop - first)*ncols,
                          offset=first*ncols*PANEL_BYTES).reshape(stop-first, ncols)
    if not np.isfinite(packs["scale"]).all() or np.any(packs["scale"] < 0):
        raise ValueError("Invalid XE82 scale")
    return packs


def parse(raw):
    if len(raw) < HEAD.size:
        raise ValueError("Short XE82 tensor")
    magic, rows, cols, tile_rows, tile_cols = HEAD.unpack_from(raw)
    if (magic, tile_rows, tile_cols) != (MAGIC, ROWS, COLS) or len(raw) != size(rows, cols):
        raise ValueError("Invalid XE82 header or byte count")
    offset = HEAD.size + (rows + cols)//8
    signs = 1 - 2 * np.unpackbits(np.frombuffer(raw[HEAD.size:offset], dtype=np.uint8),
                                   bitorder="little").astype(np.float32)
    return (rows, cols), signs[:cols], signs[cols:], memoryview(raw)[offset:]


def physical(scales, indices):
    if indices.shape != (*scales.shape, ROWS, COLS // 8):
        raise ValueError("XE82 tile/index shape differs")
    out = bytearray()
    for scale, codes in zip(scales.ravel(), indices.reshape(-1, ROWS, COLS // 8), strict=True):
        out += np.asarray(scale, dtype="<f2").tobytes()
        out += np.asarray(codes, dtype="<u2").tobytes()
    return out


class Tiles:
    def __init__(self, torch, device, dtype=None):
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.float32
        magnitudes = prototypes()
        self.book = self.tensor(magnitudes)
        self.grid = self.tensor(table_for(np.arange(65536), magnitudes))
        self.v = self.tensor(sylvester(ROWS))
        self.h = self.tensor(sylvester(COLS))

    def tensor(self, array):
        contiguous = np.ascontiguousarray(array)
        if not contiguous.flags.writeable:
            contiguous = contiguous.copy()
        return self.torch.as_tensor(contiguous, device=self.device, dtype=self.dtype)

    def plan(self, fit, su):
        nc = len(su) // COLS
        xs = self.tensor(fit).reshape(len(fit), nc, COLS).transpose(0, 1)
        xu = self.tensor(su).reshape(nc, 1, COLS)
        xr = (xs * xu) @ self.h
        covariance = xr.transpose(-1, -2) @ xr / len(fit)
        diag = covariance.diagonal(dim1=-2, dim2=-1)
        diag.add_(RIDGE * diag.mean(dim=-1, keepdim=True))
        root = self.torch.linalg.cholesky(covariance)
        factors = []
        for b in range(0, COLS-8, 8):
            stop = b + 8
            factor = self.torch.linalg.solve_triangular(
                root[:, b:stop, b:stop].transpose(-1, -2),
                root[:, stop:, b:stop].transpose(-1, -2), upper=True)
            factors.append(factor.transpose(-1, -2).contiguous())
        return factors

    def encode_slab(self, matrix, su, sv, factors):
        torch = self.torch
        nr, nc = geometry(*matrix.shape)
        z = self.tensor(matrix).reshape(nr, ROWS, nc, COLS).transpose(1, 2).contiguous()
        u = self.tensor(su).reshape(1, nc, 1, COLS)
        v = self.tensor(sv).reshape(nr, 1, ROWS, 1)
        w = self.v @ (z * u * v) @ self.h
        magnitude = w.square().mean(dim=(-2,-1)).sqrt()/1.03
        half = magnitude.to(torch.float16)
        half = torch.where((magnitude>0)&(half==0),torch.full_like(half,2**-24),half)
        if not torch.isfinite(half).all().item():
            raise ValueError("XE82 tile scale outside FP16")
        divisor = torch.where(half==0,torch.ones_like(half),half)
        w = w/divisor[...,None,None]
        out = torch.empty_like(w)
        indices = torch.empty((nr, nc, ROWS, COLS//8), dtype=torch.long, device=self.device)
        for b in reversed(range(0, COLS, 8)):
            stop = b + 8
            target = w[..., b:stop]
            if stop < COLS:
                target = target + (w[..., stop:] - out[..., stop:]) @ factors[b//8]
            flat = target.reshape(-1, 8)
            ids = torch.cat([nearest(part, self.book, torch) for part in flat.split(1536)])
            ids = ids.reshape(nr, nc, ROWS)
            indices[..., b//8] = ids
            out[..., b:stop] = self.grid[ids]
        return physical(half.cpu().numpy(), indices.cpu().numpy())

    def decode_slab(self, packs, su, sv):
        torch = self.torch
        nr, nc = packs.shape
        ids = torch.as_tensor(packs["ids"].astype(np.int64), device=self.device)
        base = self.grid[ids].reshape(nr, nc, ROWS, COLS)
        base *= self.tensor(packs["scale"].astype(np.float32))[..., None, None]
        u = self.tensor(su).reshape(1, nc, 1, COLS)
        v = self.tensor(sv).reshape(nr, 1, ROWS, 1)
        return ((self.v @ base @ self.h) * u * v).transpose(1, 2).reshape(nr*ROWS, nc*COLS)




AXES_SEED = "X30-XE82/axes/v1"
SLAB_TILES = 8
MAX_FLAT_HOST = 524 * (1 << 20)


def signs(name, shape):
    """Use the original two consecutive PCG64 draws with the HF tensor name."""
    rows, cols = shape
    seed = int.from_bytes(hashlib.sha256(f"{AXES_SEED}|{name}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return 1 - 2*rng.integers(0, 2, size=cols), 1 - 2*rng.integers(0, 2, size=rows)


def _f32(values):
    a = np.asarray(values)
    if a.dtype.kind == "u" and a.dtype.itemsize == 2:
        return (a.astype(np.uint32) << 16).view(np.float32)
    if a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Weights must be float32, float64, or uint16 BF16 words")
    return np.asarray(a, dtype=np.float32)


def _exact(stream, n):
    result = stream.read(n)
    if len(result) != n:
        raise ValueError("Truncated XE82 bytes")
    return result


class Encoder:
    """X59 codec with one physical tensor/pair of callbacks at a time.

    The source callback receives start and stop row indices and returns a NumPy
    slab spanning the input width. The sink callback receives a slice and FP32
    slab, and must consume it before the next call. For large weights, source and
    sink should be file-backed. Raw XE82 is kept separate from entropy framing.
    """
    def __init__(self, device="cuda:0"):
        import torch

        if device not in ("cpu", "cuda:0"):
            raise ValueError("Use the CPU or one visible CUDA device")
        if device=="cuda:0" and torch.cuda.device_count()!=1:
            raise ValueError("Expose exactly one CUDA device before starting XE82")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.device = device
        self.tiles = Tiles(torch, device)

    def factors(self, gram, su):
        array = np.asarray(gram)
        n = len(su)//COLS
        if array.shape != (n,COLS,COLS) or array.dtype not in (np.dtype("float32"),np.dtype("float64")):
            raise ValueError("Expected one raw FP32/FP64 Gram for each 512-column tile")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite Gram")
        torch = self.torch
        h = self.tiles.h.double()
        u = self.tiles.tensor(su).reshape(-1,COLS).double()
        m = torch.as_tensor(array,dtype=torch.float64,device=self.device)
        rotated = (h @ (m*u[:,:,None]*u[:,None,:]) @ h).float()
        diag = rotated.diagonal(dim1=-2,dim2=-1)
        diag.add_(RIDGE*diag.mean(dim=-1,keepdim=True))
        root = torch.linalg.cholesky(rotated)
        parts = []
        for low in range(0,COLS-8,8):
            stop = low+8
            value = torch.linalg.solve_triangular(root[:,low:stop,low:stop].transpose(-1,-2),
                                                  root[:,stop:,low:stop].transpose(-1,-2),upper=True)
            parts.append(value.transpose(-1,-2).contiguous())
        return parts

    def encode(self, name, shape, read, write, gram):
        """Write exact XE82 bytes to a binary stream and return the physical digest."""
        rows, cols = shape
        geometry(rows,cols)
        if SLAB_TILES*ROWS*cols*8>MAX_FLAT_HOST:
            raise ValueError("XE82 source stride would exceed the host limit")
        digest = hashlib.sha256()
        written = 0

        def emit(part):
            nonlocal written
            if write.write(part) != len(part):
                raise OSError("An XE82 physical write ended early")
            written += len(part)
            digest.update(part)

        su,sv = signs(name,shape)
        emit(prefix(su,sv))
        with self.torch.inference_mode():
            factors = self.factors(gram,su)
            for first in range(0,rows,SLAB_TILES*ROWS):
                stop = min(first+SLAB_TILES*ROWS,rows)
                weight = _f32(read(first,stop))
                if weight.shape != (stop-first,cols) or not np.isfinite(weight).all():
                    raise ValueError("Invalid source weight slab")
                emit(self.tiles.encode_slab(weight,su,sv[first:stop],factors))
        if written != size(rows,cols):
            raise ValueError("XE82 physical extent differs")
        return dict(name=name,shape=list(shape),bytes=written,sha256=digest.hexdigest())

    def decode(self, read, sink, expected_shape):
        """Cold read XE82 from a fresh binary stream; return canonical FP32 digest."""
        header = _exact(read,HEAD.size)
        magic,rows,cols,tr,tc=HEAD.unpack(header)
        if (magic,tr,tc)!=(MAGIC,ROWS,COLS) or (rows,cols)!=tuple(expected_shape):
            raise ValueError("XE82 header or requested tensor shape differs")
        nr,nc=geometry(rows,cols)
        signs_wire = _exact(read,(rows+cols)//8)
        bits = 1-2*np.unpackbits(np.frombuffer(signs_wire,np.uint8),bitorder="little").astype(np.float32)
        su,sv=bits[:cols],bits[cols:]
        native=hashlib.sha256(header+signs_wire)
        reconstructed=hashlib.sha256()
        with self.torch.inference_mode():
            for first in range(0,nr,SLAB_TILES):
                stop=min(first+SLAB_TILES,nr)
                raw=_exact(read,(stop-first)*nc*PANEL_BYTES)
                native.update(raw)
                pack=take_panels(raw,0,stop-first,nc)
                span=slice(first*ROWS,stop*ROWS)
                restored=self.tiles.decode_slab(pack,su,sv[span]).cpu().numpy()
                if not np.isfinite(restored).all():
                    raise ValueError("XE82 decoded nonfinite FP32")
                reconstructed.update(restored.astype("<f4",copy=False).tobytes())
                sink(span,restored)
        if read.read(1):
            raise ValueError("XE82 record has trailing bytes")
        return dict(shape=[rows,cols],bytes=size(rows,cols),sha256=native.hexdigest(),
                    fp32_sha256=reconstructed.hexdigest())

def _weighted_fit(residual,gram,rank,name,torch):
    # X148 uses original-coordinate ridge, unlike XE82's rotated ridge.
    rows,cols=residual.shape
    g=torch.as_tensor(gram,dtype=torch.float64,device=residual.device)
    if g.shape!=(cols//COLS,COLS,COLS) or not torch.isfinite(g).all().item():
        raise ValueError("Invalid X148 raw Gram")
    g=(g+g.transpose(-1,-2))*.5
    diag=g.diagonal(dim1=-2,dim2=-1)
    mean=diag.mean(-1,keepdim=True)
    if not (mean>0).all().item():
        raise ValueError("X148 covariance has zero diagonal mean")
    diag.add_(.02*mean)
    lower=torch.linalg.cholesky(g).float()
    whitened=(residual.reshape(rows,-1,COLS).transpose(0,1)@lower).transpose(0,1).reshape(rows,cols)
    seed=int.from_bytes(hashlib.sha256(f"x57-range-svd/v1|{name}".encode()).digest()[:8],"little")
    omega=np.random.default_rng(seed).standard_normal((cols,min(rank+16,rows,cols)),dtype=np.float32)
    sketch=torch.from_numpy(omega).to(device=residual.device,dtype=residual.dtype)
    q=torch.linalg.qr(whitened@sketch,mode="reduced").Q
    for _ in range(2):
        qt=torch.linalg.qr(whitened.T@q,mode="reduced").Q
        q=torch.linalg.qr(whitened@qt,mode="reduced").Q
    u,s,vh=torch.linalg.svd(q.T@whitened,full_matrices=False)
    roots=s[:rank].sqrt()
    left=(q@u[:,:rank])*roots
    right=vh[:rank]*roots[:,None]
    pivot=right.abs().argmax(dim=1,keepdim=True)
    axis=torch.where(right.gather(1,pivot).squeeze(1)<0,-1,1)
    right=right*axis[:,None]
    rhs=right.reshape(rank,-1,COLS).permute(1,2,0)
    answer=torch.linalg.solve_triangular(lower.transpose(-1,-2),rhs,upper=True)
    return left*axis,answer.permute(2,0,1).reshape(rank,cols)


def _int8_rows(values,*,factor=False):
    a=np.asarray(values,dtype=np.float32)
    high=np.max(np.abs(a),axis=1)
    if not np.isfinite(high).all():
        raise ValueError("Nonfinite residual rows")
    if factor:
        scale=np.maximum(high/127,2**-24).astype("<f2")
    else:
        scale=(high/127).astype("<f2")
        scale[(high>0)&(scale==0)]=np.float16(2**-24)
        scale[high==0]=1.
    if not np.isfinite(scale).all() or not np.all(scale>0):
        raise ValueError("Residual scale not representable")
    codes=np.rint(a/scale.astype(np.float32)[:,None]).clip(-127,127).astype("i1")
    return scale,codes


def vocabulary_rank(read,base,frequencies,k=2654,reserved=()):
    """Select rows by quantized residual improvement weighted by calibration frequency."""
    freq=np.asarray(frequencies)
    reserved=np.asarray(reserved,dtype=np.int64)
    rows,cols=base.shape
    if not (freq.shape==(rows,) and freq.dtype.kind in "iu" and (freq>=0).all()
            and 0<k<=rows and reserved.ndim==1 and len(np.unique(reserved))==len(reserved)<=k
            and ((0<=reserved)&(reserved<rows)).all()):
        raise ValueError("Invalid frequencies or X72 selection size")
    gain=np.empty(rows,np.float64)
    for first in range(0,rows,4*ROWS):
        stop=min(first+4*ROWS,rows)
        original=_f32(read(first,stop))
        if original.shape!=(stop-first,cols):
            raise ValueError("X72 original row slab shape differs")
        val=base[first:stop]
        residual=original-val
        scale,codes=_int8_rows(residual)
        corrected=val+codes.astype(np.float32)*scale.astype(np.float32)[:,None]
        gain[first:stop]=np.square(residual,dtype=np.float64).sum(1)-np.square(original-corrected,dtype=np.float64).sum(1)
    ids=np.arange(rows)
    energy=np.lexsort((ids,-gain))
    seen=(freq>0)&(gain>0)
    observed=energy[seen[energy]]
    observed=observed[np.lexsort((observed,-(freq[observed]*gain[observed])))]
    chosen=np.r_[observed,energy[~seen[energy]]]
    chosen=chosen[~np.isin(chosen,reserved)]
    return np.sort(np.r_[reserved,chosen[:k-len(reserved)]])


def apply_vocabulary(ids,read,base,write=None):
    """Apply the X72 INT8 rows, optionally emitting interleaved FP16/INT8 words."""
    ids=np.asarray(ids)
    rows,cols=base.shape
    if (ids.ndim!=1 or ids.dtype.kind not in "iu" or not len(ids)
            or not ((0<=ids)&(ids<rows)).all() or not (ids[1:]>ids[:-1]).all()):
        raise ValueError("X72 rows must be ordered, distinct and in range")
    dtype=np.dtype([("scale","<f2"),("codes","i1",(cols,))])
    for first in range(0,rows,4*ROWS):
        stop=min(first+4*ROWS,rows)
        a,b=np.searchsorted(ids,[first,stop])
        if a==b:
            continue
        local=ids[a:b]
        original=_f32(read(first,stop))[local-first]
        old=base[local]
        scale,codes=_int8_rows(original-old)
        if write is not None:
            payload=np.empty(len(local),dtype=dtype)
            payload["scale"],payload["codes"]=scale,codes
            blob=payload.tobytes()
            if write.write(blob)!=len(blob):
                raise OSError("Short X72 coefficient write")
        base[local]=old+codes.astype(np.float32)*scale.astype(np.float32)[:,None]
    return base
