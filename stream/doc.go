// Package stream reads, classifies and writes server-sent event streams in the
// public dialects.
//
// Framer turns an arbitrarily chunked read into complete events. ClassifyEvent
// reports what one event is on its dialect's wire — a prelude, content, a
// reasoning delta, a completion, an incomplete turn, a failure — without
// deciding retry, billing or routing policy. Observer tracks how a source
// stream ended so a caller can settle a turn, and WriteEvent, WriteDone and
// Fail emit frames in the target dialect's own shape.
package stream
