package base64x

import (
	"crypto/rand"
	"encoding/base64"
	"testing"
)

func TestEncodeDataURL(t *testing.T) {
	tests := []struct {
		mime string
		data []byte
	}{
		{"image/png", []byte("hello world")},
		{"image/jpeg", []byte("")},
		{"application/json", []byte{1, 2, 3, 4, 5}},
		{"audio/wav", make([]byte, 1024)},
	}

	for _, tt := range tests {
		got := EncodeDataURL(tt.mime, tt.data)
		want := "data:" + tt.mime + ";base64," + base64.StdEncoding.EncodeToString(tt.data)
		if got != want {
			t.Errorf("EncodeDataURL(%q, %d bytes) = %q, want %q", tt.mime, len(tt.data), got, want)
		}
	}
}

func TestEncodeDataURLAllocations(t *testing.T) {
	data := make([]byte, 512)
	mime := "image/png"

	allocs := testing.AllocsPerRun(100, func() {
		_ = EncodeDataURL(mime, data)
	})

	if allocs > 1 {
		t.Errorf("EncodeDataURL allocated %v times, want <= 1", allocs)
	}
}

func TestValidateString(t *testing.T) {
	testCases := []struct {
		input       string
		shouldValid bool
	}{
		{"", true},
		{"YQ==", true},
		{"YWI=", true},
		{"YWJj", true},
		{"YWJjZA==", true},
		{"YQ=", false},
		{"YQ===", false},
		{"Y===", false},
		{"YQ!=", false},
		{"YWJjZA=", false},
		{"======", false},
	}

	for _, tc := range testCases {
		decodedLen, ok := ValidateString(tc.input)
		if ok != tc.shouldValid {
			t.Errorf("ValidateString(%q) ok=%v, want %v", tc.input, ok, tc.shouldValid)
		}
		if ok {
			expected, err := base64.StdEncoding.DecodeString(tc.input)
			if err != nil {
				t.Fatalf("ValidateString(%q) accepted invalid standard base64", tc.input)
			}
			if decodedLen != len(expected) {
				t.Errorf("ValidateString(%q) len=%d, want %d", tc.input, decodedLen, len(expected))
			}
		}
	}
}

func TestValidateStrict(t *testing.T) {
	for size := 0; size < 100; size++ {
		buf := make([]byte, size)
		_, _ = rand.Read(buf)
		encoded := base64.StdEncoding.EncodeToString(buf)

		decLen, ok := ValidateStrict(encoded)
		if !ok || decLen != size {
			t.Fatalf("ValidateStrict failed on valid canonical base64 of size %d: ok=%v, decLen=%d", size, ok, decLen)
		}
	}

	decLen, ok := ValidateStrict("YR==")
	if ok {
		t.Errorf("ValidateStrict('YR==') expected false, got true with len=%d", decLen)
	}

	decLen, ok = ValidateStrict("YWJ=")
	if ok {
		t.Errorf("ValidateStrict('YWJ=') expected false, got true with len=%d", decLen)
	}
}

func TestValidatePermissive(t *testing.T) {
	inputs := [][]byte{
		[]byte("hello"),
		[]byte("a"),
		[]byte("ab"),
		[]byte("abc"),
		[]byte("1234567890"),
	}

	encodings := []*base64.Encoding{
		base64.StdEncoding,
		base64.RawStdEncoding,
		base64.URLEncoding,
		base64.RawURLEncoding,
	}

	for _, in := range inputs {
		for _, enc := range encodings {
			s := enc.EncodeToString(in)
			decLen, ok := ValidatePermissive(s)
			if !ok {
				t.Fatalf("ValidatePermissive failed on %q (len %d)", s, len(in))
			}
			if decLen != len(in) {
				t.Fatalf("ValidatePermissive(%q) len=%d, want %d", s, decLen, len(in))
			}
		}
	}

	invalid := []string{
		"hello world!",
		"abcde",
		"abc===",
		"====",
		"",
	}
	for _, inv := range invalid {
		if _, ok := ValidatePermissive(inv); ok {
			t.Errorf("ValidatePermissive(%q) expected false, got true", inv)
		}
	}
}

func TestValidationAllocations(t *testing.T) {
	data := make([]byte, 1024)
	_, _ = rand.Read(data)
	encoded := base64.StdEncoding.EncodeToString(data)

	allocs := testing.AllocsPerRun(100, func() {
		_, _ = ValidateString(encoded)
		_, _ = ValidateStrict(encoded)
		_, _ = ValidatePermissive(encoded)
	})

	if allocs != 0 {
		t.Errorf("Validation functions allocated %v times, want 0", allocs)
	}
}
