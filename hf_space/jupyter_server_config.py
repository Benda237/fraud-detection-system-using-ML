"""JupyterLab server config tuned for embedding inside a Hugging Face Space iframe.

HF wraps every Space in `https://huggingface.co/spaces/<id>` which iframes
`https://<id>.hf.space`. JupyterLab's default CSP (`frame-ancestors 'self'`)
blocks that, producing the "Framing ... violates the Content Security Policy"
error in the browser console.

This config widens the CSP to whitelist HF and disables the cross-origin
checks that would otherwise reject the iframed session.
"""
c = get_config()  # noqa: F821  (injected by jupyter)

c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 7860
c.ServerApp.open_browser = False
c.ServerApp.token = ""
c.ServerApp.password = ""

c.ServerApp.allow_origin = "*"
c.ServerApp.allow_remote_access = True
c.ServerApp.disable_check_xsrf = True
c.ServerApp.trust_xheaders = True
c.ServerApp.base_url = "/"

# This is the actual fix — allow HF to iframe us.
c.ServerApp.tornado_settings = {
    "headers": {
        "Content-Security-Policy": (
            "frame-ancestors 'self' "
            "https://huggingface.co https://*.huggingface.co "
            "https://hf.co https://*.hf.co "
            "https://hf.space https://*.hf.space;"
        ),
    },
}
