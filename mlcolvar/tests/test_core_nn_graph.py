import pytest
import torch

from mlcolvar.core.nn.graph.gnn import get_edge_vectors_and_lengths
from mlcolvar.core.nn.graph.radial import (
    BesselBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialEmbeddingBlock,
)


@pytest.fixture
def float64_default():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old_dtype)


def test_get_edge_vectors_and_lengths(float64_default):
    data = {
        "positions": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.07, 0.07, 0.0],
                [0.07, -0.07, 0.0],
            ]
        ),
        "edge_index": torch.tensor(
            [
                [0, 0, 1, 1, 2, 2],
                [2, 1, 0, 2, 1, 0],
            ]
        ),
        "shifts": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, -0.2, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    }

    vectors, distances = get_edge_vectors_and_lengths(
        **data,
        normalize=False,
    )

    ref_vectors = torch.tensor(
        [
            [0.07, -0.07, 0.0],
            [0.07, 0.07, 0.0],
            [-0.07, -0.07, 0.0],
            [0.0, 0.06, 0.0],
            [0.0, -0.06, 0.0],
            [-0.07, 0.07, 0.0],
        ]
    )

    ref_distances = torch.tensor(
        [
            [0.09899494936611666],
            [0.09899494936611666],
            [0.09899494936611666],
            [0.06],
            [0.06],
            [0.09899494936611666],
        ]
    )

    assert torch.allclose(vectors, ref_vectors)
    assert torch.allclose(distances, ref_distances)

    vectors, distances = get_edge_vectors_and_lengths(
        **data,
        normalize=True,
    )

    ref_vectors = torch.tensor(
        [
            [0.7071067811865476, -0.7071067811865476, 0.0],
            [0.7071067811865476, 0.7071067811865476, 0.0],
            [-0.7071067811865476, -0.7071067811865476, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-0.7071067811865476, 0.7071067811865476, 0.0],
        ]
    )

    assert torch.allclose(vectors, ref_vectors)
    assert torch.allclose(distances, ref_distances)


def test_bessel_basis(float64_default):
    reference = torch.tensor(
        [
            [0.30216178425160090, 0.6034953640555764],
            [0.29735174147757487, 0.5655966227279190],
            [0.28586135770645804, 0.47948701444265035],
            [0.26815929064765680, 0.3588671775036559],
            [0.24496326504279375, 0.22242199022921802],
            [0.21720530022724968, 0.09031904244965311],
            [0.18598678410040770, -0.019467592388889482],
            [0.15252575991598738, -0.09426610378798649],
            [0.11809918979627002, -0.12864285753339397],
            [0.08398320341397922, -0.12482336608822815],
        ]
    )

    x = torch.tensor(
        [i * 0.5 + 0.1 for i in range(10)]
    ).view(-1, 1)

    rbf = BesselBasis(6.0, n_bases=2)
    assert torch.allclose(rbf(x), reference, atol=1e-12, rtol=0)

    rbf = BesselBasis(
        6.0,
        long_range_cutoff=10.0,
        n_bases=2,
    )

    mask = torch.tensor([True, False] * 5).view(-1, 1)
    result = rbf(x, mask)

    reference_lr = torch.tensor(
        [
            [0.14047318504712697, 0.2808077400201456],
            [0.29735174147757487, 0.5655966227279190],
            [0.13771654840461342, 0.2591497039213089],
            [0.26815929064765680, 0.3588671775036559],
            [0.13052398436734916, 0.20626836096621437],
            [0.21720530022724968, 0.09031904244965311],
            [0.11931667012564413, 0.1341318339565809],
            [0.15252575991598738, -0.09426610378798649],
            [0.10474546144085174, 0.05844610427994538],
            [0.08398320341397922, -0.12482336608822815],
        ]
    )

    mask = mask.squeeze()
    assert torch.allclose(
        result[~mask],
        reference[~mask],
        atol=1e-12,
        rtol=0,
    )
    assert torch.allclose(
        result[mask],
        reference_lr[mask],
        atol=1e-12,
        rtol=0,
    )


def test_gaussian_basis(float64_default):
    reference = torch.tensor(
        [
            [0.9998611207557263, 0.6166385641763439],
            [0.9950124791926823, 0.6669768108584744],
            [0.9833348700493460, 0.7164317992468783],
            [0.9650691177896804, 0.7642281651714904],
            [0.9405880633643421, 0.8095716486678869],
            [0.9103839103891423, 0.8516705072294410],
            [0.8750517756337902, 0.8897581848801761],
            [0.8352702114112720, 0.9231163463866358],
            [0.7917795893122607, 0.9510973184771084],
            [0.7453593045429805, 0.9731449630580510],
        ]
    )

    x = torch.tensor(
        [i * 0.5 + 0.1 for i in range(10)]
    ).view(-1, 1)

    rbf = GaussianBasis(6.0, n_bases=2)
    assert torch.allclose(rbf(x), reference, atol=1e-12, rtol=0)

    rbf = GaussianBasis(
        6.0,
        long_range_cutoff=60.0,
        n_bases=2,
    )

    mask = torch.tensor([True, False] * 5).view(-1, 1)

    result = rbf(x, mask)
    assert torch.allclose(
        result[~mask.squeeze()],
        reference[~mask.squeeze()],
        atol=1e-12,
        rtol=0,
    )

    result = rbf(x * 10, mask)
    assert torch.allclose(
        result[mask.squeeze()],
        reference[mask.squeeze()],
        atol=1e-12,
        rtol=0,
    )


def test_polynomial_cutoff(float64_default):
    reference = torch.tensor(
        [
            [1.0],
            [0.9999919136092714],
            [0.9995588277320531],
            [0.9957733154296875],
            [0.9803383630544124],
            [0.9390599059360889],
            [0.85546875],
            [0.7184512221655127],
            [0.5317786922725198],
            [0.3214569091796875],
        ]
    )

    x = torch.tensor(
        [i * 0.5 for i in range(10)]
    ).view(-1, 1)

    cutoff = PolynomialCutoff(6.0)
    assert torch.allclose(
        cutoff(x),
        reference,
        atol=1e-12,
        rtol=0,
    )

    cutoff = PolynomialCutoff(6.0, 60.0)
    mask = torch.tensor([True, False] * 5).view(-1, 1)

    result = cutoff(x, mask)
    assert torch.allclose(
        result[~mask.squeeze()],
        reference[~mask.squeeze()],
        atol=1e-12,
        rtol=0,
    )

    result = cutoff(x * 10, mask)

    assert (result[~mask.squeeze()][2:] == 0).all()
    assert torch.allclose(
        result[mask.squeeze()],
        reference[mask.squeeze()],
        atol=1e-12,
        rtol=0,
    )


def test_radial_embedding_block(float64_default):
    x = torch.tensor(
        [i * 0.5 + 0.1 for i in range(10)]
    ).view(-1, 1)

    reference = torch.tensor(
        [
            [0.302161784075405670, 0.603495363703668900],
            [0.297344780473306900, 0.565583382110980900],
            [0.285645292705329600, 0.479124599728231300],
            [0.266549578182040000, 0.356712961747292670],
            [0.238761404317085600, 0.216790818528859370],
            [0.201179558989195350, 0.083655164534829570],
            [0.154832684273361420, -0.016206633178216297],
            [0.104419964978618930, -0.064535087460860160],
            [0.057909938358517744, -0.063080025890725560],
            [0.023554408472511446, -0.035008673547055544],
        ]
    )

    embedding = RadialEmbeddingBlock(6, -1.0, 2, 6)

    assert torch.allclose(
        embedding(x),
        reference,
        atol=1e-12,
        rtol=0,
    )