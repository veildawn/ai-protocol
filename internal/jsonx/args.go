package jsonx

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// ArgsValue re-renders a tool-arguments string — one JSON object — with the
// key order the string itself states, at every level. The returned bytes are
// a RawMessage, so a marshal of a body carrying them embeds them verbatim
// instead of re-sorting through a map: a body marshal compacts the bytes
// (whitespace is not semantic there), but the ORDER survives, which a
// decoded Go map cannot give back. ok is false for a string that is not
// exactly one JSON object, leaving the caller to apply its own rule for
// unusable arguments.
//
// The wire the codec folds onto renders an arguments object straight from
// the decoded value, and the reference states the keys in the caller's own
// order: every multi-key argument would come out alphabetized where the
// source stated an order. The arguments STRING is the one place the order
// survives decoding, so it is re-rendered from the string, never through a
// map. Strings are escaped the way the reference encoder escapes them — the
// ASCII escapes and \uXXXX for every code point past 0x7F — because the
// bytes are also compared directly against the reference's output.
func ArgsValue(s string) (json.RawMessage, bool) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var buf bytes.Buffer
	tok, err := dec.Token()
	if err != nil {
		return nil, false
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, false
	}
	if !writeObject(dec, &buf) {
		return nil, false
	}
	// Exactly one object and nothing after it but whitespace.
	if _, err := dec.Token(); err != io.EOF {
		return nil, false
	}
	return json.RawMessage(buf.Bytes()), true
}

// writeObject renders the object whose opening delimiter was consumed.
func writeObject(dec *json.Decoder, buf *bytes.Buffer) bool {
	buf.WriteByte('{')
	first := true
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return false
		}
		name, ok := key.(string)
		if !ok {
			return false
		}
		if !first {
			buf.WriteString(", ")
		}
		first = false
		writeString(buf, name)
		buf.WriteString(": ")
		if !writeValue(dec, buf) {
			return false
		}
	}
	if _, err := dec.Token(); err != nil { // the closing '}'
		return false
	}
	buf.WriteByte('}')
	return true
}

// writeValue renders one value: an object, an array, or a literal.
func writeValue(dec *json.Decoder, buf *bytes.Buffer) bool {
	tok, err := dec.Token()
	if err != nil {
		return false
	}
	delim, isDelim := tok.(json.Delim)
	if !isDelim {
		writeLiteral(buf, tok)
		return true
	}
	switch delim {
	case '{':
		return writeObject(dec, buf)
	case '[':
		buf.WriteByte('[')
		first := true
		for dec.More() {
			if !first {
				buf.WriteString(", ")
			}
			first = false
			if !writeValue(dec, buf) {
				return false
			}
		}
		if _, err := dec.Token(); err != nil { // the closing ']'
			return false
		}
		buf.WriteByte(']')
		return true
	default:
		return false
	}
}

// writeLiteral renders a string, number, bool or null a decoder handed back.
func writeLiteral(buf *bytes.Buffer, tok json.Token) {
	if s, ok := tok.(string); ok {
		writeString(buf, s)
		return
	}
	// Numbers keep their lexeme (UseNumber); bools and null re-marshal.
	b, _ := json.Marshal(tok)
	buf.Write(b)
}

// writeString escapes s exactly as the reference encoder does: the two JSON
// escapes, the five single-letter control escapes, \u00XX for the rest of
// the control range, and \uXXXX for every code point beyond ASCII — an
// astral code point as its surrogate pair, which is also how the reference
// spells it.
func writeString(buf *bytes.Buffer, s string) {
	buf.WriteByte('"')
	for i := 0; i < len(s); {
		c, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case c == '"' || c == '\\':
			buf.WriteByte('\\')
			buf.WriteRune(c)
		case c == '\n' || c == '\r' || c == '\t' || c == '\b' || c == '\f':
			buf.WriteByte('\\')
			switch c {
			case '\n':
				buf.WriteByte('n')
			case '\r':
				buf.WriteByte('r')
			case '\t':
				buf.WriteByte('t')
			case '\b':
				buf.WriteByte('b')
			case '\f':
				buf.WriteByte('f')
			}
		case c < 0x20:
			writeU4(buf, uint16(c))
		case c < 0x80:
			buf.WriteRune(c)
		case c <= 0xFFFF:
			writeU4(buf, uint16(c))
		default:
			r1, r2 := utf16.EncodeRune(c)
			writeU4(buf, uint16(r1))
			writeU4(buf, uint16(r2))
		}
		i += size
	}
	buf.WriteByte('"')
}

func writeU4(buf *bytes.Buffer, v uint16) {
	const hex = "0123456789abcdef"
	buf.WriteString(`\u`)
	buf.WriteByte(hex[v>>12&0xF])
	buf.WriteByte(hex[v>>8&0xF])
	buf.WriteByte(hex[v>>4&0xF])
	buf.WriteByte(hex[v&0xF])
}
