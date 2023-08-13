""" Tests for Explainer class.
"""

import pytest
import boxhed_shap


def test_wrapping_for_text_to_text_teacher_forcing_model():
    """ This tests using the Explainer class to auto wrap a masker in a text to text scenario.
    """

    transformers = pytest.importorskip("transformers")

    def f(x): # pylint: disable=unused-argument
        pass

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    wrapped_model = boxhed_shap.models.TeacherForcing(f, similarity_model=model, similarity_tokenizer=tokenizer)
    masker = boxhed_shap.maskers.Text(tokenizer, mask_token="...")

    explainer = boxhed_shap.Explainer(wrapped_model, masker, seed=1)

    assert boxhed_shap.utils.safe_isinstance(explainer.masker, "boxhed_shap.maskers.OutputComposite")

def test_wrapping_for_topk_lm_model():
    """ This tests using the Explainer class to auto wrap a masker in a language modelling scenario.
    """

    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    wrapped_model = boxhed_shap.models.TopKLM(model, tokenizer)
    masker = boxhed_shap.maskers.Text(tokenizer, mask_token="...")

    explainer = boxhed_shap.Explainer(wrapped_model, masker, seed=1)

    assert boxhed_shap.utils.safe_isinstance(explainer.masker, "boxhed_shap.maskers.FixedComposite")
