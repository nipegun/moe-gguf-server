#pragma once

#include "server-common.h"
#include "server-http.h"

#include <cstdint>
#include <string>

//
// Chats of the Web UI, kept by the server in ~/.mgs/chats/ of the user that runs mgs (in the deployed service,
// /opt/moe-gguf-server/.mgs/chats/), so every browser that opens the interface sees the same ones. Each chat is one
// JSONL file, readable only by that user, named after its creation date in UTC (2026-11-03@14-59-59.723.jsonl, with
// -2, -3... when two chats share the millisecond): its first line is {"conversation": {...}} and every other line one
// {"message": {...}}. The dates of the files (createdAt and lastModified of the conversation, timestamp of every
// message) use the native date format of the project (2026-11-03@14-59-59.723, UTC); the API gives and takes them in
// milliseconds since the epoch, computed from the native format.
//
// GET    <api prefix>/chats                 {"version", "conversations"}, the last modified first
// GET    <api prefix>/chats/version         {"version"}: it changes with every write, so open browsers reload
// GET    <api prefix>/chats/search?q=&limit= the newest message of the user or the assistant of every chat whose
//                                            content holds every word of q, without regard to case or accents
// GET    <api prefix>/chats/:id_chat        {"conversation", "messages"}
// POST   <api prefix>/chats/:id_chat        {"conversation"?, "messages"?, "deleted"?, "replace"?}: creates the chat
//                                            or saves its changes (messages are added or replaced by id, "deleted"
//                                            lists ids to remove, "replace" makes "messages" the whole list)
// DELETE <api prefix>/chats/:id_chat        removes the chat and its file
//
// A chat saved without a valid number (1 to 99, not held by another chat) gets the lowest free one, 0 when the 99
// are taken; one saved without createdAt gets the current time.
//

// The native date of the project for pMs milliseconds since the epoch: UTC, 2026-11-03@14-59-59.723.
std::string fChatsNativeDate(int64_t pMs);

// The milliseconds since the epoch of a native date, or -1 when pDate is not one.
int64_t fChatsDateToMs(const std::string &pDate);

server_http_context::handler_t fChatsMakeListHandler();
server_http_context::handler_t fChatsMakeVersionHandler();
server_http_context::handler_t fChatsMakeSearchHandler();
server_http_context::handler_t fChatsMakeGetHandler();
server_http_context::handler_t fChatsMakePostHandler();
server_http_context::handler_t fChatsMakeDeleteHandler();
