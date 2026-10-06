# Muryllo plotts — Backend para Render

Este é o backend real do GOES Explorer 98, preparado para o Muryllo plotts.

## Deploy no Render

1. Coloque esta pasta em um repositório GitHub.
2. No Render, crie um **Web Service** a partir do repositório.
3. O Render detectará o `Dockerfile`.
4. O serviço deve usar `/status` como health check.
5. Depois do deploy, o Render fornecerá uma URL HTTPS.
6. Essa URL será usada pelo frontend Muryllo plotts como `BACKEND_URL`.

## Endpoints

- `GET /`
- `GET /status`
- `GET /satellites`
- `GET /bands`
- `POST /generate`
- `POST /generate-gif`

## Compatibilidade

O backend continua aceitando `duration_hours` na API original e também aceita
`duration_minutes`, que é o formato usado pelo frontend atual.

CORS está habilitado para permitir que o frontend hospedado separadamente faça
as requisições. Depois que o domínio definitivo do frontend existir, pode-se
restringir `allow_origins` a esse domínio.
