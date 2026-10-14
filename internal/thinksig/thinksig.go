// Package thinksig carries the signature a fold gives a thinking block it
// rebuilt onto the Messages wire from a dialect that has no signatures.
//
// The Messages wire spells an unsigned thinking block with signature null (or
// "" in a stream's content_block_start), and Messages-protocol clients treat
// exactly that spelling as "no signature": they store nothing and replay the
// thinking as a plain TEXT block on the next turn. Down a translating gateway
// that is not a cosmetic loss — the replayed text block lands between the
// tool_use blocks of the same turn, and the fold of the NEXT request has to
// carry a shape it never produced. The observed breakage (a reasoning model
// behind a Responses upstream, 2026-10-14): every multi-turn tool loop died on
// its second request because the first turn's thinking came back unsigned.
//
// A fold's source dialect has no Messages signatures at all — nothing on a
// fold path can verify one — so the honest signature here is a clearly marked
// stand-in, not a forged one. A backend that DID verify signatures would
// reject it; no such backend exists behind a fold, whose upstream never signs.
package thinksig

// Synthetic is the stand-in signature for a fold-produced thinking block:
// base64 of "unsigned-thinking". Non-empty so clients keep the block as
// thinking across replay, and self-describing to anyone who base64-decodes it
// while debugging.
const Synthetic = "dW5zaWduZWQtdGhpbmtpbmc="

// Or returns the signature a Messages thinking block should carry: a real one
// when the source dialect had it, the stand-in when it did not.
func Or(real string) string {
	if real != "" {
		return real
	}
	return Synthetic
}
