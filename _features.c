/*
 * _features.c — C extension for fast character-class feature extraction.
 *
 * Directly reads CPython's internal Unicode buffer (PEP 393 compact form)
 * without any copy.  Returns (cjk, letter, digit, punct, space, word) as
 * a tuple of ints.
 *
 * Build:  python setup.py build_ext --inplace
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *
extract_features(PyObject *self, PyObject *args)
{
    PyObject *text;
    if (!PyArg_ParseTuple(args, "U", &text))   /* "U" = Unicode string */
        return NULL;

    /* Ensure internal buffer is ready (should already be for interned strings) */
    if (PyUnicode_READY(text) < 0)
        return NULL;

    Py_ssize_t n  = PyUnicode_GET_LENGTH(text);
    int kind      = PyUnicode_KIND(text);
    const void *data = PyUnicode_DATA(text);

    long long cjk = 0, letter = 0, digit = 0, space = 0, word = 0;
    int in_word = 0;

    for (Py_ssize_t i = 0; i < n; i++) {
        Py_UCS4 c = PyUnicode_READ(kind, data, i);

        if (c <= 0x7f) {
            /* ASCII fast path */
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                letter++;
                if (!in_word) { word++; in_word = 1; }
            } else if (c >= '0' && c <= '9') {
                digit++;
                if (!in_word) { word++; in_word = 1; }
            } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                space++;
                in_word = 0;
            } else {
                /* ASCII punct/symbol */
                if (!in_word) { word++; in_word = 1; }
            }
        } else {
            /* Non-ASCII */
            if ((c >= 0x4e00 && c <= 0x9fff) ||
                (c >= 0x3000 && c <= 0x303f) ||
                (c >= 0xff00 && c <= 0xffef)) {
                cjk++;
            }
            /* Other non-ASCII whitespace */
            else if (c == 0x85 || c == 0xa0 ||
                     (c >= 0x2000 && c <= 0x200b)) {
                space++;
                in_word = 0;
                continue;
            }
            /* Everything else (emoji, Cyrillic, etc.) → punct (implicit) */
            if (!in_word) { word++; in_word = 1; }
        }
    }

    long long punct = (long long)n - cjk - letter - digit - space;

    return Py_BuildValue("(LLLLLL)", cjk, letter, digit, punct, space, word);
}


static PyMethodDef methods[] = {
    {"extract_features", extract_features, METH_VARARGS,
     "extract_features(text) -> (cjk, letter, digit, punct, space, word)\n"
     "Single-pass character classification on CPython internal buffer.\n"
     "Coefficients and scoring are handled by the Python layer."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_features",
    "Fast character-class feature extraction (C extension)", -1, methods
};

PyMODINIT_FUNC PyInit__features(void) {
    return PyModule_Create(&module);
}
