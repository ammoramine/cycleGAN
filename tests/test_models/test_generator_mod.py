"""
    testing the model
"""

def test_if_shape_is_the_same(random_batch_one,model):
    print(f"batch size : {random_batch_one.shape}")
    out = model(random_batch_one)
    print(f"output batch size : {out.shape}")
    assert out.shape == random_batch_one.shape