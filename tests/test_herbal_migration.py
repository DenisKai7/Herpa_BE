import pytest

from app.core.deduplicate_herbs import (
    normalize_scientific_name,
    select_canonical_node,
)


def test_normalize_scientific_name():
    assert normalize_scientific_name("Kaempferia galanga L.") == "kaempferia galanga l"
    assert normalize_scientific_name("Curcuma longa L.") == "curcuma longa l"
    assert normalize_scientific_name("Curcuma zedoaria (Christm.) Roscoe") == "curcuma zedoaria christm roscoe"
    assert normalize_scientific_name(None) == ""


def test_select_canonical_node():
    nodes = [
        {
            "element_id": "4:eaac2ba4-a233-49f2-bbd5-7a128da7efc0:988",
            "id": "HRB-533-KEN",
            "commonName": "Kencur",
            "latinName": "Kaempferia galanga L.",
            "speciesNumber": 533,
        },
        {
            "element_id": "4:eaac2ba4-a233-49f2-bbd5-7a128da7efc0:261",
            "id": "HRB-005-KEN",
            "commonName": "Kencur",
            "latinName": "Kaempferia galanga L.",
            "speciesNumber": 5,
        },
        {
            "element_id": "4:eaac2ba4-a233-49f2-bbd5-7a128da7efc0:1195",
            "id": "HRB-740-KEN",
            "commonName": "Kencur",
            "latinName": "Kaempferia galanga L.",
            "speciesNumber": 740,
        }
    ]

    canonical = select_canonical_node(nodes)
    assert canonical["id"] == "HRB-005-KEN"
    assert canonical["speciesNumber"] == 5
