"""Input validation and fuzz tests for the HTTP contract.

A serving endpoint that accepts NaN, Inf, an empty payload or a 20 MB body is
not a commercial endpoint, it is a way to lose a worker.  These tests pin the
edge behaviour: every invalid request is a 4xx with a JSON body, never a 500,
never a hang, and never a truncated success.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from henri import api_server

client = TestClient(api_server.app)


def post_raw(path: str, body: str):
    """Send a body verbatim.

    ``httpx`` refuses to serialise NaN or Infinity, so a non-finite payload has
    to be sent as raw text.  That is also how a hostile client sends it: the
    server's JSON parser is what accepts or rejects the literal.
    """
    return client.post(path, content=body, headers={"content-type": "application/json"})


# -- valid baseline ------------------------------------------------------

def test_rank_accepts_two_options() -> None:
    response = client.post(
        "/v1/reason/rank",
        json={"question": "Which function sorts a list?", "options": ["sort_list", "reverse_list"]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


# -- reason/rank ---------------------------------------------------------

def test_rank_rejects_a_single_option() -> None:
    response = client.post("/v1/reason/rank", json={"question": "q", "options": ["only"]})
    assert response.status_code == 422


def test_rank_rejects_an_empty_question() -> None:
    response = client.post("/v1/reason/rank", json={"question": "   ", "options": ["a", "b"]})
    assert response.status_code == 422


def test_rank_rejects_a_nan_weight() -> None:
    body = '{"question": "q", "options": ["a", "b"], "channel_weight": NaN}'
    response = post_raw("/v1/reason/rank", body)
    assert response.status_code == 422


def test_rank_rejects_an_infinite_weight() -> None:
    for literal in ("Infinity", "-Infinity"):
        body = f'{{"question": "q", "options": ["a", "b"], "channel_weight": {literal}}}'
        response = post_raw("/v1/reason/rank", body)
        assert response.status_code == 422, literal


def test_rank_rejects_a_hundred_thousand_options() -> None:
    payload = {"question": "q", "options": [f"option_{index}" for index in range(100_000)]}
    response = client.post("/v1/reason/rank", json=payload)
    assert response.status_code == 422


def test_rank_rejects_an_oversized_question() -> None:
    payload = {"question": "x" * (api_server.MAX_TEXT_CHARS + 1), "options": ["a", "b"]}
    response = client.post("/v1/reason/rank", json=payload)
    assert response.status_code == 422


# -- robotics/step -------------------------------------------------------

def test_robotics_rejects_empty_sensors() -> None:
    response = client.post("/v1/robotics/step", json={"sensor_values": []})
    assert response.status_code == 422


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_robotics_rejects_non_finite_sensors(literal: str) -> None:
    body = f'{{"sensor_values": [0.1, {literal}, 0.2]}}'
    response = post_raw("/v1/robotics/step", body)
    assert response.status_code == 422


def test_robotics_rejects_a_hundred_thousand_sensors() -> None:
    response = client.post(
        "/v1/robotics/step",
        json={"sensor_values": [0.0] * 100_000},
    )
    assert response.status_code == 422


# -- code/solve ----------------------------------------------------------

def test_code_solve_rejects_an_empty_spec() -> None:
    response = client.post("/v1/code/solve", json={"spec": "  ", "candidates": ["def f(): pass"]})
    assert response.status_code == 422


def test_code_solve_rejects_an_oversized_spec() -> None:
    response = client.post(
        "/v1/code/solve",
        json={"spec": "x" * (api_server.MAX_TEXT_CHARS + 1), "candidates": ["def f(): pass"]},
    )
    assert response.status_code == 422


def test_code_solve_rejects_too_many_candidates() -> None:
    response = client.post(
        "/v1/code/solve",
        json={"spec": "add", "candidates": ["def f(a,b): return a+b"] * (api_server.MAX_ITEMS + 1)},
    )
    assert response.status_code == 422


# -- cartridges/write ----------------------------------------------------

def test_cartridges_rejects_an_empty_tenant() -> None:
    response = client.post(
        "/v1/cartridges/write", json={"tenant_id": " ", "keys": ["k"], "values": ["v"]}
    )
    assert response.status_code == 422


def test_cartridges_rejects_unequal_lengths() -> None:
    response = client.post(
        "/v1/cartridges/write", json={"tenant_id": "t", "keys": ["a", "b"], "values": ["v"]}
    )
    assert response.status_code == 422


def test_cartridges_rejects_too_many_pairs() -> None:
    count = api_server.MAX_ITEMS + 1
    response = client.post(
        "/v1/cartridges/write",
        json={"tenant_id": "t", "keys": [f"k{index}" for index in range(count)],
              "values": [f"v{index}" for index in range(count)]},
    )
    assert response.status_code == 422


# -- contract shape ------------------------------------------------------

def test_error_bodies_are_json_and_stable() -> None:
    response = client.post("/v1/reason/rank", json={"question": "q", "options": ["only"]})
    payload = response.json()
    assert response.status_code == 422
    assert isinstance(payload, dict)
    assert payload, "an error response must explain itself"
