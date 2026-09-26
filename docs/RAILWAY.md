# Despliegue en Railway

- Proyecto: `hermes-trading` (workspace "moebiussc's Projects")
- Entorno: `production`; servicio: `hermes-trading`
- Builder: RAILPACK (con `Dockerfile` en el repo)
- Región: sfo, 1 réplica
- Dominio público: hermes-trading-production-c90e.up.railway.app (puerto 8080)
- Volumen montado en `/app/state` (estado en ejecución; no está en el repo)

## Variables de entorno requeridas (solo nombres, definir los valores en Railway)

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `HERMES_DASHBOARD_PASSWORD`
- `HERMES_STATE_TOKEN`
- `LLM_API_KEY`
- `PORT`
