// Package messagesresponses converts between the Messages dialect and the
// Responses dialect: request bodies, response bodies and SSE streams.
//
// Neither wire is expressed in terms of the other, so the package carries the
// mapping for each direction itself rather than routing through a third
// dialect.
package messagesresponses
