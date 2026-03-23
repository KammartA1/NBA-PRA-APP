"""
streamlit_app — Pure-frontend Streamlit application package.

Every page module exposes a ``render()`` function.  The main entry point
(``app_refactored.py``) imports pages and calls ``render()`` inside the
appropriate tab.

Design rules
------------
* NO computation in Streamlit code -- delegate to service modules.
* ``st.session_state`` is used ONLY for transient UI state (form values
  being edited, toggle states, temporary display data).
* All persistent data is read from / written to the database via service
  functions in the ``services/`` package.
* If Streamlit crashes and restarts, ZERO data is lost.
"""
