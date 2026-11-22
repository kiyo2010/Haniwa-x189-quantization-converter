"""Small, CPU-only verification of the standalone XE82 path."""
import hashlib
from io import BytesIO

import numpy as np

from .e8 import Encoder, ROWS, COLS, apply_vocabulary, prototypes, table_for, vocabulary_rank


def fixture():
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    book=table_for(np.arange(65536,dtype=np.uint16),prototypes()).astype("<f4")
    digest=hashlib.sha256(book.tobytes()).hexdigest()
    assert digest=="69038254fded6881558c6369a333e7e5a9160a1425321c215a1cef0278cbea7a"
    # Known input and independent cold output in the complete XE82 format.
    rng=np.random.default_rng(59001)
    original=rng.normal(0,.02,(ROWS,COLS)).astype(np.float32)
    bits=(original.view(np.uint32)>>16).astype("<u2")
    reference=(bits.astype(np.uint32)<<16).view(np.float32)
    gram=np.eye(COLS,dtype=np.float64)[None]
    codec=Encoder("cpu")
    with torch.inference_mode():
        points=codec.tiles.tensor(book[np.array([0,124,299,65535])])
        from .e8 import nearest
        codes=nearest(points,codec.tiles.book,torch).numpy()
        np.testing.assert_array_equal(book[codes],points.numpy())
    stream=BytesIO()
    card=codec.encode("synthetic",bits.shape,lambda a,b:bits[a:b],stream,gram)
    restore=np.empty_like(reference)
    cold=codec.decode(BytesIO(stream.getvalue()),restore.__setitem__,bits.shape)
    assert card["sha256"]==cold["sha256"] and card["bytes"]==len(stream.getvalue())
    assert np.linalg.norm(restore-reference)<np.linalg.norm(reference)
    for wire in (stream.getvalue()[:-1],stream.getvalue()+b"0",b"XXXX"+stream.getvalue()[4:]):
        try:
            codec.decode(BytesIO(wire),restore.__setitem__,bits.shape)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid XE82 escaped independent decoder")
    source=np.array([[1,2],[2,4],[3,6],[4,8]],dtype=np.float32)
    baseline=np.zeros_like(source)
    scores=vocabulary_rank(lambda a,b:source[a:b],baseline,np.array([8,9,0,0]),k=2)
    np.testing.assert_array_equal(scores,[0,1])
    before=np.square(source-baseline).sum()
    apply_vocabulary(scores,lambda a,b:source[a:b],baseline)
    assert np.square(source-baseline).sum()<before
    from .e8 import _weighted_fit, _int8_rows
    residual=np.outer(rng.normal(size=16),rng.normal(size=COLS)).astype(np.float32)
    with torch.inference_mode():
        left,right=_weighted_fit(torch.from_numpy(residual),gram,1,"rank-one-fixture",torch)
        np.testing.assert_allclose((left@right).numpy(),residual,rtol=3e-5,atol=2e-6)
    scale,codes=_int8_rows(np.array([[0,-127,127],[0,0,0]],np.float32),factor=True)
    assert scale.tolist()==[1.,2**-24] and codes.tolist()==[[0,-127,127],[0,0,0]]
    return dict(e8p_all_codepoints_sha256=digest,wire_sha256=card["sha256"],decoded_fp32_sha256=cold["fp32_sha256"],
                corrupted_header_or_extent_rejected=True,x72_frequency_order=True,rank_factor_roundtrip=True,
                actual_backend="cpu")


if __name__=="__main__":
    print(fixture())
