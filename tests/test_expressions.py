import pytest

from voicedm.expressions import has_speech, history_text, speech_text


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
    assert speech_text(action, sounds=True) == tag


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
        "Don't *laugh* at me.",
        "",
    ],
)
def test_everything_else_is_left_as_it_is(text):
    assert speech_text(text, sounds=True) == text


def test_only_the_action_is_replaced():
    said = "*laughs* I can't believe it. *smiles* You did it! *sighs*"
    assert (
        speech_text(said, sounds=True) == "[laugh] I can't believe it. *smiles* You did it! [sigh]"
    )


def test_action_naming_two_sounds_becomes_both_tags_once():
    assert speech_text("*laughs and coughs*", sounds=True) == "[laugh] [cough]"
    assert speech_text("*laughs, laughing*", sounds=True) == "[laugh]"


def test_nothing_changes_unless_asked():
    said = "*laughs* I can't. *looks at you with a mix of amusement and pride* Well."
    assert speech_text(said) == said


@pytest.mark.parametrize(
    ("said", "spoken"),
    [
        # Narration: four words or more.
        (
            "*looks at you with a mix of amusement and a bit of pride* My work is going well.",
            "My work is going well.",
        ),
        ("I see. *her eyes light up* Tell me more.", "I see. Tell me more."),
        ("That's all. *leans back in the chair*", "That's all."),
        # Shorter, with a verb that actions are made of.
        ("*smiles* Hello.", "Hello."),
        ("*nods slowly* Yes.", "Yes."),
        ("*looks away* Maybe.", "Maybe."),
        ("Okay. *grinning widely* Let's go.", "Okay. Let's go."),
        ("Well, *shrugs* I guess so.", "Well, I guess so."),
        ("Sure thing *winks*", "Sure thing"),
        ("*nodding* Right.", "Right."),
        ("*sigh* Fine.", "Fine."),
        ("Okay. *wink*", "Okay."),
        # The punctuation around a removed action.
        ("Fine *shrugs*.", "Fine."),
        ("*smiles*. Hello.", "Hello."),
        ("Yes. *nods*. Go on.", "Yes. Go on."),
        ("*smiles* *waves* Hi!", "Hi!"),
        # Sounds are actions too when they are not voiced.
        ("*laughs* Good one.", "Good one."),
    ],
)
def test_actions_are_left_out_of_the_speech(said, spoken):
    assert speech_text(said, silent=True) == spoken


@pytest.mark.parametrize(
    "said",
    [
        "I *never* said that.",
        "That is *really* good.",
        "I love you *so much*.",
        "Not *right now*, please.",
        "I *don't* know.",
        "A *well-known* fact.",
        "It was *three times* as big.",
        "That *looks* great.",
        "*Look* at me.",
        "Stop *staring* at it.",
        "Don't *laugh* at me.",
        "**smiles at you warmly today**",
        "2*3*4",
        "*5*",
        "No asterisks here.",
    ],
)
def test_emphasis_is_still_spoken(said):
    assert speech_text(said, silent=True) == said


@pytest.mark.parametrize(
    ("said", "spoken"),
    [
        ("*laughs* Good one. *smiles*", "[laugh] Good one."),
        ("*smiles* Sure. *sighs*", "Sure. [sigh]"),
        # A long action is silent anyway, so the sound it names can be heard.
        ("*looks at you and laughs softly at the idea* No.", "[laugh] No."),
        ("*walks to the window and looks out* Hm.", "Hm."),
        ("I *never* laugh.", "I *never* laugh."),
    ],
)
def test_sounds_and_silent_actions_together(said, spoken):
    assert speech_text(said, sounds=True, silent=True) == spoken


def test_long_action_naming_a_sound_is_read_out_when_actions_are_not_silent():
    said = "*looks at you and laughs softly at the idea* No."
    assert speech_text(said, sounds=True) == said


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Hello.", True), ("[laugh]", True), ("5", True), ("", False), (" ", False), ("...", False)],
)
def test_has_speech(text, expected):
    assert has_speech(text) is expected


LOOKS = "*looks at you with a mix of amusement and a bit of pride*"


@pytest.mark.parametrize(
    ("written", "shown"),
    [
        (f"{LOOKS} My work is going well.", "My work is going well."),
        ("*smiles* Hello. *nods*", "Hello."),
        ("*laughs* Good one.", "Good one."),
        ("I *never* said that.", "I *never* said that."),
        ("No actions here.", "No actions here."),
        # Nothing but actions: shown as written rather than as an empty reply.
        (LOOKS, LOOKS),
        ("*nods* *smiles*", "*nods* *smiles*"),
    ],
)
def test_history_is_shown_without_actions(written, shown):
    assert history_text(written) == shown


@pytest.mark.parametrize(
    ("written", "shown"),
    [
        (f"{LOOKS} *laughs* Good one. *smiles*", "*laughs* Good one."),
        ("*sighs softly* Fine. *shrugs*", "*sighs softly* Fine."),
        # A long action is narration even when it names a sound.
        ("*looks at you and laughs softly at the idea* No.", "No."),
        ("*laughs*", "*laughs*"),
    ],
)
def test_history_can_keep_the_sound_actions(written, shown):
    assert history_text(written, keep_sounds=True) == shown
