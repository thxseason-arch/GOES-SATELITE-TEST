# GOES Explorer 98 — Backend v2

Backend separado do notebook original. O arquivo `Banco de dados SªT ams.ipynb` é mantido apenas como referência e não é modificado.

## Rodar no Google Colab

```bash
pip install -r requirements.txt
```

```python
import nest_asyncio, uvicorn, threading
nest_asyncio.apply()
threading.Thread(target=lambda: uvicorn.run('app.main:app',host='0.0.0.0',port=8000), daemon=True).start()
```

Endpoints:
- GET `/`
- GET `/status`
- GET `/satellites`
- GET `/bands`
- POST `/generate`
- POST `/generate-gif`

O backend reaproveita as funções centrais do notebook para busca ABI, GLM, paletas e renderização.
