from pathlib import Path

path = Path('.env')
text = path.read_text()
client_id = input('Paste X OAuth 2.0 Client ID: ').strip()
client_secret = input('Paste X OAuth 2.0 Client Secret, or press Enter for public PKCE app: ').strip()
text = text.replace('X_CLIENT_ID=\n', f'X_CLIENT_ID={client_id}\n')
text = text.replace('X_CLIENT_SECRET=\n', f'X_CLIENT_SECRET={client_secret}\n')
path.write_text(text)
print('Wrote .env')