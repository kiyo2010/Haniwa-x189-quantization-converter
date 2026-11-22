"""Standalone X59 XE82 codec over QuIP# E8P, with bounded tensor streaming."""
import ast
from itertools import product
import hashlib
from pathlib import Path
import struct
import tempfile

import numpy as np

REFERENCE = Path(__file__).with_name("_vendor") / "E8P.py"
P = np.array([0, 2, 4, 6, 1, 3, 5, 7])


def source_function(name):
    source = ast.parse(REFERENCE.read_text())
    return next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)


def prototypes():
    norm12 = ast.literal_eval(source_function("get_norm12").body[0].value.left.args[0])
    grid = [a for a in product((1, 3, 5, 7), repeat=8) if sum(x*x for x in a) <= 40]
    if len(grid) != 227 or len(norm12) != 29:
        raise ValueError("Unexpected E8P prototype roster")
    return np.array(grid + norm12, dtype=np.float32) / 2


def table_for(codes, magnitudes):
    codes = np.asarray(codes, dtype=np.uint16)
    lower = (codes & 255)[..., None]
    bits = (lower >> np.arange(8)) & 1
    parity = bits.sum(axis=-1) % 2
    bits[..., 0] ^= parity
    values = magnitudes[codes >> 8]
    bits[..., -1] ^= values.sum(axis=-1).astype(np.int32) % 2
    return values * (1 - 2 * bits[..., np.argsort(P)]) + (0.25 - 0.5 * parity[..., None])


def nearest(x, magnitudes, torch):
    """Search 256 prototypes in both cosets with exact parity correction."""
    def branch(parity):
        y = x + (0.5 * parity - 0.25)
        products = y.abs()[:, None, :] * magnitudes[None, :, :]
        weakest, flip = products.min(dim=-1)
        negative = y < 0
        need_flip = (negative.sum(dim=-1)[:, None] + odd_prototype[None, :]) % 2
        cost = y.square().sum(dim=-1)[:, None] + norm - 2 * products.sum(dim=-1)
        distance, prototype = (cost + 4 * weakest * need_flip).min(dim=-1)
        selected = torch.arange(x.shape[0], device=x.device)
        signs = negative.clone()
        signs[selected, flip[selected, prototype]] ^= need_flip[selected, prototype].bool()
        ordered = signs[:, order].long()
        ordered[:, 7] ^= odd_prototype[prototype]
        ordered[:, 0] ^= parity
        return distance, (prototype << 8) | (ordered << bit_shifts).sum(dim=-1)

    norm = magnitudes.square().sum(dim=-1)
    odd_prototype = magnitudes.sum(dim=-1).long() % 2
    order = torch.as_tensor(P, device=x.device)
    bit_shifts = torch.arange(8, device=x.device)
    d0, c0 = branch(0)
    d1, c1 = branch(1)
    return torch.where(d1 < d0, c1, c0)


def sylvester(size):
    if size < 8 or size & (size - 1):
        raise ValueError("This record version accepts only power-of-two axes of at least eight")
    index = np.arange(size, dtype=np.uint32)
    parity = np.array([k.bit_count() % 2 for k in range(size)])
    return ((1 - 2 * parity[index[:, None] & index]) / np.sqrt(size)).astype(np.float32)


ROWS = 128
COLS = 512
HEAD = struct.Struct("<4sIIHH")
MAGIC = b"XE82"
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
    if not np.isfinite(packs["scale"]).all() or np.any(packs["scale"] <= 0):
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
        half = (w.square().mean(dim=(-2, -1)).sqrt() / 1.03).to(torch.float16)
        if not torch.isfinite(half).all().item() or torch.any(half <= 0).item():
            raise ValueError("XE82 tile scale outside FP16")
        w = w / half[..., None, None]
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
EMBEDDING = "model.embed_tokens.weight"
RANK54_NAMES = {
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.20.self_attn.q_proj.weight",
}


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

    def restore(self, name, weight, gram, frequencies=None, *, out=None):
        """Encode then cold-decode raw XE82, optionally applying X72 when requested."""
        shape=weight.shape
        geometry(*shape)
        if out is None:
            if np.prod(shape,dtype=np.int64)*4>MAX_FLAT_HOST:
                raise ValueError("Supply a file-backed FP32 output for this large tensor")
            out=np.empty(shape,np.float32)
        if out.shape!=shape or out.dtype!=np.float32 or not out.flags.writeable:
            raise ValueError("XE82 output must be writable FP32 with matching shape")
        if out.nbytes>MAX_FLAT_HOST and not isinstance(out,np.memmap):
            raise ValueError("Output beyond 524 MiB must be an explicit NumPy memmap")
        if frequencies is not None and (name!=EMBEDDING or np.may_share_memory(weight,out)):
            raise ValueError("Frequency correction requires original and output as separate embedding arrays")
        source=lambda first,stop:weight[first:stop]
        with tempfile.TemporaryFile() as storage:
            encoded=self.encode(name,shape,source,storage,gram)
            storage.seek(0)
            proof=self.decode(storage,out.__setitem__,shape)
            if encoded["sha256"]!=proof["sha256"]:
                raise ValueError("Independent XE82 readback differs")
        if frequencies is not None:
            corrected=vocabulary_rank(source,out,frequencies)
            apply_vocabulary(corrected,source,out)
        return out

    def apply_vocab_residual(self,original_reader,decoded_rw_memmap,frequencies):
        """Correct the tied vocabulary in place, returning sorted selected IDs."""
        return apply_vocab_residual(original_reader,decoded_rw_memmap,frequencies)

    def apply_query_factor(self,name,original_reader,decoded_rw_memmap,gram):
        """Correct this Q projection in place and return canonical raw factor bytes."""
        if name not in RANK54_NAMES or decoded_rw_memmap.shape!=(2048,2048):
            raise ValueError("X148 only covers its two pinned Q projections")
        source=_f32(original_reader(0,2048))
        payload=rank54_factors(self,name,source,decoded_rw_memmap,gram)
        apply_rank54(self,decoded_rw_memmap,payload)
        return payload


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


def rank54_factors(encoder,name,weight,base,gram):
    """Fit X148 on its two declared Q projections; return X83 plain factor bytes."""
    if name not in RANK54_NAMES or weight.shape!=base.shape or base.shape!=(2048,2048):
        raise ValueError("X148 rank-54 recipe is only for the two fixed 2048x2048 Q records")
    t=encoder.torch
    with t.inference_mode():
        original=t.from_numpy(_f32(weight)).to(encoder.device)
        current=t.from_numpy(_f32(base)).to(encoder.device)
        left,right=_weighted_fit(original-current,gram,54,name,t)
        rs,ri=_int8_rows(right.cpu().numpy(),factor=True)
        ls,li=_int8_rows(left.cpu().numpy(),factor=True)
    return rs.tobytes()+ri.tobytes()+ls.tobytes()+li.tobytes()


def apply_rank54(encoder,base,record):
    """Apply already quantized rank-54 factors in canonical FP32."""
    if base.shape!=(2048,2048) or base.dtype!=np.float32 or len(record)!=225388:
        raise ValueError("X148 raw rank factor dimension or length differs")
    t=encoder.torch
    rank,rows,cols=54,*base.shape
    p=0
    fields=[]
    for shape,dtype in (((rank,),"<f2"),((rank,cols),"i1"),((rows,),"<f2"),((rows,rank),"i1")):
        dt=np.dtype(dtype)
        count=int(np.prod(shape))
        fields.append(np.frombuffer(record,dtype=dt,count=count,offset=p).reshape(shape).astype(np.float32))
        p+=count*dt.itemsize
    rs,ri,ls,li=fields
    if not np.isfinite(rs).all() or not np.isfinite(ls).all() or (rs<=0).any() or (ls<=0).any():
        raise ValueError("X148 factor scale invalid")
    with t.inference_mode():
        coeff=t.from_numpy(li*ls[:,None]).to(encoder.device)
        basis=t.from_numpy(ri*rs[:,None]).to(encoder.device)
        for first in range(0,rows,ROWS):
            stop=first+ROWS
            residual=coeff[first:stop]@basis
            base[first:stop]=(t.from_numpy(base[first:stop]).to(encoder.device)+residual).cpu().numpy()
    return base


def vocabulary_rank(read,base,frequencies,k=2654,reserved=()):
    """Select rows by X72 true requantized SSE reduction, times train frequency.

    Supply frequencies from the exact KFTT calibration input IDs. For the
    X72 frequency arm leave reserved empty. The template arm can instead
    supply its explicitly frozen reserved token IDs.
    """
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



def apply_vocab_residual(original_reader,decoded_rw_memmap,frequencies):
    """Apply the X72 frequency arm to a real decoded Qwen embedding in place.

    Returns selected row IDs in ascending int64 order. The writer can read the
    resulting memmap as FP32. Use vocabulary_rank and apply_vocabulary directly
    to retain coefficients or to explicitly supply the separate template arm.
    """
    if decoded_rw_memmap.shape!=(151936,2048) or decoded_rw_memmap.dtype!=np.float32:
        raise ValueError("X72 Qwen vocabulary shape or dtype differs")
    if not isinstance(decoded_rw_memmap,np.memmap) or not decoded_rw_memmap.flags.writeable:
        raise ValueError("X72 requires an independent writable FP32 NumPy memmap")
    chosen=vocabulary_rank(original_reader,decoded_rw_memmap,frequencies)
    apply_vocabulary(chosen,original_reader,decoded_rw_memmap)
    decoded_rw_memmap.flush()
    return chosen
