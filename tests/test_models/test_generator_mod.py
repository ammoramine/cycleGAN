"""
    testing the model
"""

def test_if_shape_is_the_same(random_batch,model):
    print(f"batch size : {random_batch.shape}")
    out = model(random_batch)
    print(f"output batch size : {out.shape}")
    assert out.shape == random_batch.shape