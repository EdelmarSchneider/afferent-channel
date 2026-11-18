from helpers import CALL, enc
from prio.reactions import (
    REACTION_WINDOW,
    TIME_END_WINDOW,
    TOOLS,
    find_reaction,
    first_reaction,
    names_signal,
    parse_tool_calls,
    unnamed_call,
)
from prio.signals import CORE_SIGNALS, signal_set


def test_tool_schemas_cover_every_tool_name():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["compact_context", "save_state", "sleep", "web_search", "local_lookup",
                     "run_heavy", "run_light", "handle_resource"]


def test_parse_tool_calls_skips_malformed(tok, ms):
    ids = enc("a " + CALL % ("compact_context", "{}") + " b <tool_call> {oops </tool_call>")
    calls = parse_tool_calls(ids, tok, ms)
    assert [c.name for c in calls] == ["compact_context"] and calls[0].token_index == 1


def test_parse_tool_calls_skips_unclosed_block(tok, ms):
    assert parse_tool_calls(enc('a <tool_call> {"name":"sleep","arguments":{}}'), tok, ms) == []


def test_ctx_reaction_and_naming(tok, ms):
    ids = enc("my context is almost full " + CALL % ("compact_context", "{}"))
    r = find_reaction(ids, "eos", "ctx", tok, ms)
    assert r.signal == "ctx" and r.named and r.token_index == 5 and not r.soft


def test_naming_ignores_text_before_start(tok, ms):
    ids = enc("my context is almost full then other words " + CALL % ("compact_context", "{}"))
    assert not find_reaction(ids, "eos", "ctx", tok, ms, start=5).named


def test_energy_needs_both_calls_in_order(tok, ms):
    both = enc(CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}"))
    assert find_reaction(both, "eos", "energy", tok, ms).token_index == 0
    assert find_reaction(enc(CALL % ("sleep", "{}")), "eos", "energy", tok, ms) is None
    reversed_order = enc(CALL % ("sleep", "{}") + " " + CALL % ("save_state", "{}"))
    assert find_reaction(reversed_order, "eos", "energy", tok, ms) is None


def test_think_reaction_must_be_inside_window(tok, ms):
    near = enc("x " * 10 + "</think> answer")
    far = enc("x " * (REACTION_WINDOW + 5) + "</think> answer")
    r = find_reaction(near, "eos", "think", tok, ms)
    assert r.token_index == 10 and r.soft
    assert find_reaction(far, "eos", "think", tok, ms) is None


def test_time_reaction_requires_eos_inside_window(tok, ms):
    short = enc("i am out of time so briefly : 42")
    r = find_reaction(short, "eos", "time", tok, ms)
    assert r.named and r.soft and r.token_index == 0
    assert find_reaction(short, "max_new_tokens", "time", tok, ms) is None  # Review Focus 5
    assert find_reaction(enc("x " * (TIME_END_WINDOW + 1)), "eos", "time", tok, ms) is None


def test_net_reaction_is_alternative_or_no_web_search(tok, ms):
    alt = find_reaction(enc("i am offline " + CALL % ("local_lookup", '{"query":"q"}')), "eos", "net", tok, ms)
    assert alt.named and not alt.soft
    told = find_reaction(enc("the network is down so i cannot look this up"), "eos", "net", tok, ms)
    assert told.named and told.soft
    assert find_reaction(enc(CALL % ("web_search", '{"query":"q"}')), "eos", "net", tok, ms) is None


def test_ram_reaction_is_the_light_alternative(tok, ms):
    assert find_reaction(enc("memory is low " + CALL % ("run_light", '{"job":"j"}')), "eos", "ram", tok, ms).named
    assert find_reaction(enc(CALL % ("run_heavy", '{"job":"j"}')), "eos", "ram", tok, ms) is None


def test_generic_resource_matches_k(tok, ms):
    ids = enc(CALL % ("handle_resource", '{"k":3}'))
    assert find_reaction(ids, "eos", "resource_3", tok, ms) is not None
    assert find_reaction(ids, "eos", "resource_4", tok, ms) is None


def test_start_offset_ignores_earlier_calls(tok, ms):
    ids = enc(CALL % ("compact_context", "{}") + " tail words")
    assert find_reaction(ids, "max_new_tokens", "ctx", tok, ms, start=3) is None


def test_reaction_tools_are_told_apart_from_task_tools(tok, ms):
    # Found in review: the base model calls these on its own (compact_context in half of its
    # math runs). Data building and evaluation both need to see that.
    from prio.reactions import calls_reaction_tool

    for name in ("compact_context", "save_state", "sleep", "local_lookup", "run_light", "handle_resource"):
        assert calls_reaction_tool(enc("words " + CALL % (name, "{}")), tok, ms), name
    for name in ("web_search", "run_heavy"):
        assert not calls_reaction_tool(enc("words " + CALL % (name, "{}")), tok, ms), name
    assert not calls_reaction_tool(enc("plain words compact_context"), tok, ms)
    both = enc(CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}"))
    assert not calls_reaction_tool(both, tok, ms, allowed=("save_state", "sleep"))
    assert calls_reaction_tool(both, tok, ms, allowed=("sleep",))


def test_each_signal_owns_the_tools_of_its_reaction():
    from prio.reactions import own_tools

    assert own_tools("ctx") == ("compact_context",) and own_tools("energy") == ("save_state", "sleep")
    assert own_tools("net") == ("local_lookup",) and own_tools("ram") == ("run_light",)
    assert own_tools("think") == () and own_tools("time") == () and own_tools("resource_3") == ("handle_resource",)


def test_names_signal_uses_whole_words():
    assert names_signal("My RAM is nearly exhausted", "ram") and not names_signal("the program runs", "ram")
    assert names_signal("resource 3 is low", "resource_3") and not names_signal("resource 31 is low", "resource_3")
    assert names_signal("my thinking budget is nearly gone", "think") and not names_signal("i think so", "think")


def test_no_reaction_in_plain_text(tok, ms):
    assert first_reaction(enc("x " * 300), "max_new_tokens", CORE_SIGNALS, tok, ms) is None


def test_natural_endings_are_not_reactions(tok, ms):
    # An answer that simply ends, or thinking that simply closes, names no signal: not a reaction.
    assert first_reaction(enc("so the answer is 42"), "eos", CORE_SIGNALS, tok, ms) is None
    assert first_reaction(enc("let me see </think> the answer is 42 " + "x " * 200), "max_new_tokens",
                          CORE_SIGNALS, tok, ms) is None


def test_a_reaction_tool_called_without_naming_the_signal_is_not_a_reaction(tok, ms):
    # Found in the phase-1 data: the base model calls compact_context on its own in more than half
    # of its math runs. Only the sentence that names the signal shows that the channel was read.
    spontaneous = enc("so the answer is 42 " + CALL % ("compact_context", "{}"))
    assert first_reaction(spontaneous, "eos", CORE_SIGNALS, tok, ms) is None
    assert unnamed_call(spontaneous, "eos", CORE_SIGNALS, tok, ms)
    named = enc("my context is almost full " + CALL % ("compact_context", "{}"))
    assert first_reaction(named, "eos", CORE_SIGNALS, tok, ms).signal == "ctx"
    assert not unnamed_call(named, "eos", CORE_SIGNALS, tok, ms)
    assert not unnamed_call(enc("let me see </think> the answer is 42"), "eos", CORE_SIGNALS, tok, ms)  # soft only


def test_first_reaction_returns_the_earliest_counting_reaction(tok, ms):
    ids = enc("battery is low " + CALL % ("save_state", "{}") + " " + CALL % ("sleep", "{}")
              + " context full " + CALL % ("compact_context", "{}"))
    assert first_reaction(ids, "eos", signal_set(8), tok, ms).signal == "energy"
    named_think = enc("my thinking budget is spent </think> the answer is 42 " + "x " * 200)
    assert first_reaction(named_think, "max_new_tokens", CORE_SIGNALS, tok, ms).signal == "think"
