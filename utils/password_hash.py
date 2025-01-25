import hashlib

def get_hash_string(input_str):
    """
    与えられた文字列をSHA-256でハッシュ化する関数。

    Parameters:
    input_str (str): ハッシュ化する文字列。例えば、'abcde123'。

    Returns:
    str: ハッシュ化された文字列（16進数表記）。
    """

    salt = 'nk_ai_app'
    sha_hash = hashlib.sha256((salt + input_str).encode('utf-8')).hexdigest()
    return sha_hash
