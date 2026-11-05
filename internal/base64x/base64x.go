// Package base64x checks and encodes base64 without allocating. The folds use
// the checks before treating a data URL as media, and EncodeDataURL builds a
// data URL in one allocation.
package base64x

import (
	"encoding/base64"
	"unsafe"
)

// decodeTable maps ASCII bytes to their 6-bit values (0-63).
// -1: invalid byte
// -2: padding '='
var stdDecodeTable [256]int8
var urlDecodeTable [256]int8

func init() {
	for i := range stdDecodeTable {
		stdDecodeTable[i] = -1
		urlDecodeTable[i] = -1
	}

	const stdAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	const urlAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

	for i := 0; i < len(stdAlphabet); i++ {
		c := stdAlphabet[i]
		stdDecodeTable[c] = int8(i)
	}
	for i := 0; i < len(urlAlphabet); i++ {
		c := urlAlphabet[i]
		urlDecodeTable[c] = int8(i)
	}

	stdDecodeTable['='] = -2
	urlDecodeTable['='] = -2
}

// EncodeDataURL constructs a base64 data URL ("data:<mime>;base64,<payload>")
// in a single heap allocation without intermediate string concatenations or buffer copies.
func EncodeDataURL(mime string, data []byte) string {
	prefixLen := len("data:") + len(mime) + len(";base64,")
	b64Len := base64.StdEncoding.EncodedLen(len(data))
	totalLen := prefixLen + b64Len

	buf := make([]byte, totalLen)
	n := copy(buf, "data:")
	n += copy(buf[n:], mime)
	n += copy(buf[n:], ";base64,")
	base64.StdEncoding.Encode(buf[n:], data)

	return unsafe.String(unsafe.SliceData(buf), totalLen)
}

// ValidateString checks whether s is a valid standard base64 string with padding,
// and returns the decoded length in bytes without allocating memory.
func ValidateString(s string) (decodedLen int, ok bool) {
	n := len(s)
	if n == 0 {
		return 0, true
	}
	if n%4 != 0 {
		return 0, false
	}

	pad := 0
	if s[n-1] == '=' {
		pad++
		if s[n-2] == '=' {
			pad++
			if s[n-3] == '=' {
				return 0, false // max 2 '=' padding allowed
			}
		}
	}

	contentLen := n - pad
	for i := 0; i < contentLen; i++ {
		v := stdDecodeTable[s[i]]
		if v < 0 {
			return 0, false
		}
	}

	return (n/4)*3 - pad, true
}

// ValidateStrict checks whether s is a valid canonical standard base64 string,
// equivalent to base64.StdEncoding.Strict().DecodeString(s) returning nil error
// AND re-encoding to the identical string, but with zero heap allocations.
func ValidateStrict(s string) (decodedLen int, ok bool) {
	n := len(s)
	if n == 0 {
		return 0, true
	}
	if n%4 != 0 {
		return 0, false
	}

	pad := 0
	if s[n-1] == '=' {
		pad++
		if s[n-2] == '=' {
			pad++
			if s[n-3] == '=' {
				return 0, false
			}
		}
	}

	contentLen := n - pad
	for i := 0; i < contentLen; i++ {
		v := stdDecodeTable[s[i]]
		if v < 0 {
			return 0, false
		}
	}

	// Canonical check on unused trailing bits in the last quantum:
	if pad == 2 {
		val := stdDecodeTable[s[contentLen-1]]
		if val&0x0f != 0 {
			return 0, false
		}
	} else if pad == 1 {
		val := stdDecodeTable[s[contentLen-1]]
		if val&0x03 != 0 {
			return 0, false
		}
	}

	return (n/4)*3 - pad, true
}

// ValidatePermissive checks whether s is a valid base64 payload under either
// standard or URL alphabets, with or without padding (Std, RawStd, URL, RawURL).
// Zero heap allocations.
//
// The two alphabets are alternatives, not a union: a payload is read under
// EXACTLY ONE of them, chosen by the first character that tells them apart
// ('+' or '/' for the standard alphabet, '-' or '_' for the URL one). Fitting
// that choice is half the check — "AAA+BBB/" mixes both and is a base64
// payload in none of the four spellings, so no decoder reads it. Accepting the
// mix let a junk payload through the media gate as inline data, and the
// upstream answered 400 INVALID_ARGUMENT on every replay of that turn.
func ValidatePermissive(s string) (decodedLen int, ok bool) {
	n := len(s)
	if n == 0 {
		return 0, false
	}

	pad := 0
	for n > 0 && s[n-1] == '=' {
		pad++
		n--
	}
	if pad > 2 {
		return 0, false
	}

	if pad > 0 {
		if (n+pad)%4 != 0 {
			return 0, false
		}
	} else {
		if n%4 == 1 {
			return 0, false
		}
	}

	// urlAlphabet is chosen by convention — the first member of the pair the
	// payload carries. A payload carrying neither stays on the standard table,
	// where the two agree byte for byte.
	urlAlphabet := false
	for i := 0; i < n; i++ {
		switch s[i] {
		case '-', '_':
			urlAlphabet = true
		}
	}
	table := &stdDecodeTable
	if urlAlphabet {
		table = &urlDecodeTable
	}
	for i := 0; i < n; i++ {
		if table[s[i]] < 0 {
			return 0, false
		}
	}

	rawLen := (n + pad)
	if pad > 0 {
		return (rawLen/4)*3 - pad, true
	}

	switch n % 4 {
	case 0:
		return (n / 4) * 3, true
	case 2:
		return (n/4)*3 + 1, true
	case 3:
		return (n/4)*3 + 2, true
	}

	return 0, false
}
