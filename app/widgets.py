"""Click-only selectors, used instead of st.selectbox (whose search field lets the user type)."""
import streamlit as st


def _keep_selection(key: str, last: str) -> None:
    """Clicking the selected button clears the choice; put the previous choice back instead."""
    if st.session_state[key] is None:
        st.session_state[key] = st.session_state[last]
    else:
        st.session_state[last] = st.session_state[key]


def pick(label: str, options: list[str], default: str, key: str, where=st, pills: bool = False, **kwargs) -> str:
    """Single choice as buttons (pills=True wraps long lists over several rows); never returns None."""
    last = f"{key}__last"
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(last, default)
    st.session_state.setdefault(last, st.session_state[key])
    widget = where.pills if pills else where.segmented_control
    return widget(label, options, key=key, on_change=_keep_selection, args=(key, last), **kwargs)
