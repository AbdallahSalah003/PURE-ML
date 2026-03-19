import re
import time
import socket
import ssl
import json
import dns.resolver
import whois
import requests
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from datetime import datetime


# ── Feature order (must match training exactly) ───────────────────────────────

FEATURE_ORDER = [
    'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
    'qty_equal_url', 'qty_at_url', 'qty_and_url', 'qty_percent_url',
    'qty_tld_url', 'length_url', 'qty_dot_domain', 'qty_hyphen_domain',
    'qty_vowels_domain', 'domain_length', 'qty_dot_directory',
    'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
    'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory',
    'qty_and_directory', 'qty_exclamation_directory', 'qty_space_directory',
    'qty_tilde_directory', 'qty_comma_directory', 'qty_plus_directory',
    'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
    'qty_percent_directory', 'directory_length', 'qty_dot_file',
    'qty_hyphen_file', 'qty_underline_file', 'qty_slash_file',
    'qty_questionmark_file', 'qty_equal_file', 'qty_at_file', 'qty_and_file',
    'qty_exclamation_file', 'qty_space_file', 'qty_tilde_file',
    'qty_comma_file', 'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file',
    'qty_dollar_file', 'qty_percent_file', 'file_length', 'qty_dot_params',
    'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
    'qty_questionmark_params', 'qty_equal_params', 'qty_at_params',
    'qty_and_params', 'qty_exclamation_params', 'qty_space_params',
    'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
    'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
    'qty_percent_params', 'params_length', 'tld_present_params',
    'qty_params', 'email_in_url', 'time_response', 'domain_spf', 'asn_ip',
    'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
    'qty_nameservers', 'qty_mx_servers', 'ttl_hostname',
    'tls_ssl_certificate', 'qty_redirects',
]

DYNAMIC_FEATURES = [
    'time_response', 'domain_spf', 'asn_ip',
    'time_domain_activation', 'time_domain_expiration',
    'qty_ip_resolved', 'qty_nameservers', 'qty_mx_servers',
    'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects',
]

# ── Load medians at startup ───────────────────────────────────────────────────

_MEDIANS_PATH = Path("data/processed/feature_medians.json")
_MEDIANS: dict = {}

if _MEDIANS_PATH.exists():
    with open(_MEDIANS_PATH) as f:
        _MEDIANS = json.load(f)
    print(f"Loaded medians for {len(_MEDIANS)} dynamic features")
else:
    print(f"WARNING: {_MEDIANS_PATH} not found — dynamic features will use -1")


def _impute(feature_name: str, value: float) -> float:
    """
    Replace -1 (failed network call) with the training median.
    This keeps the feature distribution close to what the model learned.
    If no median available, return 0 as a safe fallback.
    """
    if value == -1:
        return _MEDIANS.get(feature_name, 0)
    return value


# ── URL parser ────────────────────────────────────────────────────────────────

def _parse_url(url: str) -> dict:
    """
    Split URL into domain, directory, file, params.

    https://example.com/path/to/file.html?a=1&b=2
      domain    → example.com
      directory → /path/to/
      file      → file.html
      params    → a=1&b=2
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    path   = parsed.path
    params = parsed.query

    if "/" in path:
        last_slash = path.rfind("/")
        directory  = path[:last_slash + 1]
        file_part  = path[last_slash + 1:]
    else:
        directory = path
        file_part = ""

    return {
        "domain":    domain,
        "directory": directory,
        "file":      file_part,
        "params":    params,
        "parsed":    parsed,
    }


# ── Character counter ─────────────────────────────────────────────────────────

def _count(text: str) -> dict:
    return {
        "dot":          text.count("."),
        "hyphen":       text.count("-"),
        "underline":    text.count("_"),
        "slash":        text.count("/"),
        "questionmark": text.count("?"),
        "equal":        text.count("="),
        "at":           text.count("@"),
        "and":          text.count("&"),
        "exclamation":  text.count("!"),
        "space":        text.count(" ") + text.count("%20"),
        "tilde":        text.count("~"),
        "comma":        text.count(","),
        "plus":         text.count("+"),
        "asterisk":     text.count("*"),
        "hashtag":      text.count("#"),
        "dollar":       text.count("$"),
        "percent":      text.count("%"),
    }


# ── Static features ───────────────────────────────────────────────────────────

def extract_static_features(url: str) -> dict:
    parts  = _parse_url(url)
    domain = parts["domain"]
    direc  = parts["directory"]
    file_  = parts["file"]
    params = parts["params"]

    # TLD
    domain_clean = domain.split(":")[0]             # strip port
    tld          = domain_clean.split(".")[-1] if "." in domain_clean else ""

    # Character counts per URL section
    uc = _count(url)
    dc = _count(domain_clean)
    rc = _count(direc)
    fc = _count(file_)
    pc = _count(params)

    # Email pattern in URL
    email_re    = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    email_in_url = 1 if re.search(email_re, url) else 0

    # TLD present in params
    tld_in_params = 1 if (tld and tld.lower() in params.lower()) else 0

    # Number of query parameters
    qty_params = len(parse_qs(params))

    return {
        # URL level
        "qty_dot_url":                uc["dot"],
        "qty_hyphen_url":             uc["hyphen"],
        "qty_underline_url":          uc["underline"],
        "qty_slash_url":              uc["slash"],
        "qty_equal_url":              uc["equal"],
        "qty_at_url":                 uc["at"],
        "qty_and_url":                uc["and"],
        "qty_percent_url":            uc["percent"],
        "qty_tld_url":                len(tld),
        "length_url":                 len(url),

        # Domain level
        "qty_dot_domain":             dc["dot"],
        "qty_hyphen_domain":          dc["hyphen"],
        "qty_vowels_domain":          sum(1 for c in domain_clean.lower()
                                         if c in "aeiou"),
        "domain_length":              len(domain_clean),

        # Directory level
        "qty_dot_directory":          rc["dot"],
        "qty_hyphen_directory":       rc["hyphen"],
        "qty_underline_directory":    rc["underline"],
        "qty_slash_directory":        rc["slash"],
        "qty_questionmark_directory": rc["questionmark"],
        "qty_equal_directory":        rc["equal"],
        "qty_at_directory":           rc["at"],
        "qty_and_directory":          rc["and"],
        "qty_exclamation_directory":  rc["exclamation"],
        "qty_space_directory":        rc["space"],
        "qty_tilde_directory":        rc["tilde"],
        "qty_comma_directory":        rc["comma"],
        "qty_plus_directory":         rc["plus"],
        "qty_asterisk_directory":     rc["asterisk"],
        "qty_hashtag_directory":      rc["hashtag"],
        "qty_dollar_directory":       rc["dollar"],
        "qty_percent_directory":      rc["percent"],
        "directory_length":           len(direc),

        # File level
        "qty_dot_file":               fc["dot"],
        "qty_hyphen_file":            fc["hyphen"],
        "qty_underline_file":         fc["underline"],
        "qty_slash_file":             fc["slash"],
        "qty_questionmark_file":      fc["questionmark"],
        "qty_equal_file":             fc["equal"],
        "qty_at_file":                fc["at"],
        "qty_and_file":               fc["and"],
        "qty_exclamation_file":       fc["exclamation"],
        "qty_space_file":             fc["space"],
        "qty_tilde_file":             fc["tilde"],
        "qty_comma_file":             fc["comma"],
        "qty_plus_file":              fc["plus"],
        "qty_asterisk_file":          fc["asterisk"],
        "qty_hashtag_file":           fc["hashtag"],
        "qty_dollar_file":            fc["dollar"],
        "qty_percent_file":           fc["percent"],
        "file_length":                len(file_),

        # Params level
        "qty_dot_params":             pc["dot"],
        "qty_hyphen_params":          pc["hyphen"],
        "qty_underline_params":       pc["underline"],
        "qty_slash_params":           pc["slash"],
        "qty_questionmark_params":    pc["questionmark"],
        "qty_equal_params":           pc["equal"],
        "qty_at_params":              pc["at"],
        "qty_and_params":             pc["and"],
        "qty_exclamation_params":     pc["exclamation"],
        "qty_space_params":           pc["space"],
        "qty_tilde_params":           pc["tilde"],
        "qty_comma_params":           pc["comma"],
        "qty_plus_params":            pc["plus"],
        "qty_asterisk_params":        pc["asterisk"],
        "qty_hashtag_params":         pc["hashtag"],
        "qty_dollar_params":          pc["dollar"],
        "qty_percent_params":         pc["percent"],
        "params_length":              len(params),
        "tld_present_params":         tld_in_params,
        "qty_params":                 qty_params,
        "email_in_url":               email_in_url,
    }


# ── Dynamic feature helpers ───────────────────────────────────────────────────

def _time_response(domain: str, timeout: int = 3) -> float:
    try:
        # Use dns.resolver instead of socket — much faster and more reliable
        start   = time.perf_counter()
        answers = dns.resolver.resolve(domain, "A", lifetime=timeout)
        elapsed = time.perf_counter() - start
        return round(elapsed, 6)
    except Exception:
        try:
            # Fallback to socket
            start = time.perf_counter()
            socket.setdefaulttimeout(timeout)
            socket.gethostbyname(domain)
            return round(time.perf_counter() - start, 6)
        except Exception:
            return -1


def _domain_spf(domain: str) -> int:
    root = _get_root_domain(domain)   # ← use root, not subdomain
    try:
        answers = dns.resolver.resolve(root, "TXT", lifetime=5)
        for r in answers:
            if "v=spf1" in r.to_text().lower():
                return 1
        return 0
    except Exception:
        return -1


def _asn_ip(domain: str) -> int:
    try:
        ip = socket.gethostbyname(domain)
        s  = socket.socket()
        s.settimeout(5)
        s.connect(("whois.cymru.com", 43))
        s.send(f" -v {ip}\r\n".encode())
        response = s.recv(4096).decode()
        s.close()
        for line in response.splitlines():
            if line and not line.startswith("AS"):
                parts = line.split("|")
                if parts:
                    asn = parts[0].strip()
                    if asn.isdigit():
                        return int(asn)
        return -1
    except Exception:
        return -1

def _get_root_domain(domain: str) -> str:
    """Strip subdomains — WHOIS needs root domain only."""
    parts = domain.split(".")
    return ".".join(parts[-2:]) if len(parts) > 2 else domain

def _time_domain_activation(domain: str) -> int:
    root = _get_root_domain(domain)
    try:
        w       = whois.whois(root)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return -1
        if isinstance(created, str):
            from dateutil import parser as dp
            created = dp.parse(created)
        if not isinstance(created, datetime):
            return -1

        # Fix: strip timezone info to make both naive
        if created.tzinfo is not None:
            created = created.replace(tzinfo=None)

        return max(0, (datetime.now() - created).days)

    except Exception as e:
        print(f"    WHOIS activation failed ({root}): {type(e).__name__}: {e}")
        return -1


def _time_domain_expiration(domain: str) -> int:
    root = _get_root_domain(domain)
    try:
        w       = whois.whois(root)
        expires = w.expiration_date
        if isinstance(expires, list):
            expires = expires[0]
        if expires is None:
            return -1
        if isinstance(expires, str):
            from dateutil import parser as dp
            expires = dp.parse(expires)
        if not isinstance(expires, datetime):
            return -1

        # Fix: strip timezone info
        if expires.tzinfo is not None:
            expires = expires.replace(tzinfo=None)

        return max(0, (expires - datetime.now()).days)

    except Exception as e:
        print(f"    WHOIS expiration failed ({root}): {type(e).__name__}: {e}")
        return -1

def _qty_ip_resolved(domain: str) -> int:
    try:
        results = dns.resolver.resolve(domain, "A", lifetime=5)
        return len(results)
    except Exception:
        return -1


def _qty_nameservers(domain: str) -> int:
    try:
        results = dns.resolver.resolve(domain, "NS", lifetime=5)
        return len(results)
    except Exception:
        return -1


def _qty_mx_servers(domain: str) -> int:
    try:
        results = dns.resolver.resolve(domain, "MX", lifetime=5)
        return len(results)
    except Exception:
        return -1


def _ttl_hostname(domain: str) -> int:
    try:
        results = dns.resolver.resolve(domain, "A", lifetime=5)
        return int(results.rrset.ttl)
    except Exception:
        return -1


def _tls_ssl_certificate(domain: str, timeout: int) -> int:
    try:
        ctx  = ssl.create_default_context()
        conn = ctx.wrap_socket(
            socket.socket(socket.AF_INET),
            server_hostname=domain
        )
        conn.settimeout(timeout)
        conn.connect((domain, 443))
        cert = conn.getpeercert()
        conn.close()
        # Check certificate is not expired
        expire_str = cert.get("notAfter", "")
        if expire_str:
            expire_dt = datetime.strptime(expire_str, "%b %d %H:%M:%S %Y %Z")
            if expire_dt > datetime.now():
                return 1
        return 0
    except Exception:
        return 0


def _qty_redirects(url: str, timeout: int) -> int:
    try:
        response = requests.get(
            url,
            timeout         = timeout,
            allow_redirects = True,
            headers         = {"User-Agent": "Mozilla/5.0"},
        )
        return len(response.history)
    except Exception:
        return -1


# ── Dynamic features (with imputation) ───────────────────────────────────────

def extract_dynamic_features(url: str, timeout: int = 5) -> dict:
    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0]   # strip port

    print(f"  Fetching dynamic features for: {domain}")

    raw = {
        "time_response":          _time_response(domain, timeout),
        "domain_spf":             _domain_spf(domain),
        "asn_ip":                 _asn_ip(domain),
        "time_domain_activation": _time_domain_activation(domain),
        "time_domain_expiration": _time_domain_expiration(domain),
        "qty_ip_resolved":        _qty_ip_resolved(domain),
        "qty_nameservers":        _qty_nameservers(domain),
        "qty_mx_servers":         _qty_mx_servers(domain),
        "ttl_hostname":           _ttl_hostname(domain),
        "tls_ssl_certificate":    _tls_ssl_certificate(domain, timeout),
        "qty_redirects":          _qty_redirects(url, timeout),
    }

    # Log what succeeded and what failed
    failed    = [k for k, v in raw.items() if v == -1]
    succeeded = [k for k, v in raw.items() if v != -1]
    print(f"  Dynamic OK  ({len(succeeded)}): {succeeded}")
    print(f"  Dynamic FAIL({len(failed)}):  {failed} → replaced with medians")

    # Impute failures with training medians
    imputed = {k: _impute(k, v) for k, v in raw.items()}
    return imputed


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_features(url: str) -> np.ndarray:
    """
    Full pipeline: URL string → numpy array (1, 81)
    in the exact feature order the model was trained on.
    """
    print(f"\nExtracting features for: {url}")

    static  = extract_static_features(url)
    dynamic = extract_dynamic_features(url)

    all_features = {**static, **dynamic}

    # Verify all features present
    missing = [f for f in FEATURE_ORDER if f not in all_features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    vector = np.array(
        [all_features[feat] for feat in FEATURE_ORDER],
        dtype=np.float64
    )

    print(f"  Feature vector shape: {vector.shape}")
    print(f"  Feature vector: {dict(zip(FEATURE_ORDER, vector))}\n")

    return vector.reshape(1, -1)