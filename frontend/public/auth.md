# auth.md

## Overview
Authentication and registration guide for agents using Furina ML.

## Discovery
- OAuth Protected Resource Metadata: https://furina.nevatal.id/.well-known/oauth-protected-resource
- OAuth 2.0 Authorization Server Metadata: https://furina.nevatal.id/.well-known/oauth-authorization-server
- OpenID Connect Discovery: https://furina.nevatal.id/.well-known/openid-configuration
- API Catalog: https://furina.nevatal.id/.well-known/api-catalog
- Agent Card: https://furina.nevatal.id/.well-known/agent-card.json

## Registration
Public content can be read without credentials. Agents that need write
access register at https://furina.nevatal.id/api/agent/register and receive bearer credentials.
