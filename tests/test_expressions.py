import pytest

from voicedm.expressions import with_sound_tags


@pytest.mark.parametrize(
    ("action", "tag"),
    [
        ("*laughs*", "[laugh]"),
        ("*laughing*", "[laugh]"),
        ("*giggles*", "[laugh]"),
        ("*chuckles*", "[chuckle]"),
        ("*snickering*", "[chuckle]"),
        ("*sigh*", "[sigh]"),
        ("*Sighs*", "[sigh]"),
        ("*gasps*", "[gasp]"),
        ("*coughs*", "[cough]"),
        ("*sniffles*", "[sniff]"),
        ("*sniffs*", "[sniff]"),
        ("*groans*", "[groan]"),
        ("*moans*", "[groan]"),
        ("*shushes*", "[shush]"),
        ("*shhh*", "[shush]"),
        ("*clears throat*", "[clear throat]"),
        ("*clears her throat*", "[clear throat]"),
        ("*clearing his throat*", "[clear throat]"),
        ("*laughs softly*", "[laugh]"),
        ("*lets out a sigh*", "[sigh]"),
        ("*sighs.*", "[sigh]"),
    ],
)
def test_action_naming_a_sound_becomes_its_tag(action, tag):
    assert with_sound_tags(action) == tag


@pytest.mark.parametrize(
    "text",
    [
        "*smiles*",
        "*nods slowly*",
        "*walks to the window*",
        "That is *really* good.",
        "**laughs**",
        "2*3*4",
        "* laughs *",
        "laughs",
        "[laugh]",
        "*clears the table*",
        "*throat*",
        "*show*",
        "*sighs and slowly walks over to the window*",
        "a * b and c * laughs",
        "*laughs\nloudly*",
        "",
    ],
)
def test_everything_else_is_left_as_it_is(text):
    assert with_sound_tags(text) == text


def test_only_the_action_is_replaced():
    said = "*laughs* I can't believe it. *smiles* You did it! *sighs*"
    assert with_sound_tags(said) == "[laugh] I can't believe it. *smiles* You did it! [sigh]"


def test_action_naming_two_sounds_becomes_both_tags_once():
    assert with_sound_tags("*laughs and coughs*") == "[laugh] [cough]"
    assert with_sound_tags("*laughs, laughing*") == "[laugh]"
