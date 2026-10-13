# How to Start and Deploy the Reforestation Platform

This guide provides step-by-step instructions for running the project locally in development mode and deploying it to production across multiple cloud environments.

---

## 1. Local Development Setup

### 1.1 Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.14 compatible)
- **Node.js**: Version 18+ and **npm** (only needed if modifying frontend source code; production static assets are already pre-compiled)
- **Git**

### 1.2 Step-by-Step Local Startup

#### Step 1: Clone and Enter the Repository
```bash
git clone <repository-url>
cd Reforestation
```

#### Step 2: Create and Activate a Python Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Run the Machine Learning Pipeline (Optional if outputs exist)
The project comes with generated artifacts in `ml_outputs/`. To re-train all models and re-generate predictions:
```bash
python3 ml_pipeline.py
```
*Runtime*: ~4-5 minutes (trains 13 algorithms with 5-fold cross-validation and SMOTE analysis).

#### Step 5: Start the Flask Backend Server
```bash
python3 app.py
```
The server will start at `http://127.0.0.1:5000/`. Open this URL in any modern browser to view the interactive dashboard.

---

## 2. Frontend Development (Modifying UI)

The frontend source is located in `frontend/`. 

```bash
cd frontend
npm install
npm run dev
```
Vite dev server will launch at `http://localhost:5173/`, with API requests automatically proxied to Flask on port 5000.

To build updated frontend assets into Flask's `static/` directory:
```bash
npm run build
```

---

## 3. Production Deployment Options

---

### Option A: Docker Deployment (Recommended)

The easiest and most reproducible way to run the platform in production.

#### 1. Build the Docker Image
```bash
docker build -t reforestation-ai .
```

#### 2. Run the Container
```bash
docker run -d -p 5000:5000 --name reforestation-app --restart always reforestation-ai
```
The app is immediately accessible at `http://<server-ip>:5000`.

#### Or use Docker Compose:
```bash
docker-compose up -d
```

---

### Option B: Deploying to Render (Free / Paid)

1. Create a new account on [Render.com](https://render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub/GitLab repository.
4. Configure the settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4`
5. Click **Create Web Service**.

---

### Option C: Deploying to Railway

1. Install Railway CLI or connect via the [Railway Dashboard](https://railway.app).
2. Create a **New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Railway automatically detects `requirements.txt` and `Procfile`.
4. The service will bind to `$PORT` automatically.

---

### Option D: Deploying to a Linux VPS (Ubuntu / Debian / RHEL)

For deploying to an AWS EC2 instance, DigitalOcean Droplet, Linode, or bare-metal server:

#### 1. System Setup
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv nginx
```

#### 2. Setup Project Directory
```bash
cd /var/www
sudo git clone <repo-url> reforestation
cd reforestation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. Configure Systemd Service
Create `/etc/systemd/system/reforestation.service`:
```ini
[Unit]
Description=Gunicorn instance to serve Reforestation Intelligence
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/reforestation
Environment="PATH=/var/www/reforestation/venv/bin"
ExecStart=/var/www/reforestation/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl start reforestation
sudo systemctl enable reforestation
```

#### 4. Configure Nginx as Reverse Proxy
Edit `/etc/nginx/sites-available/reforestation`:
```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /var/www/reforestation/static/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }
}
```

Enable and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/reforestation /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. Install SSL Certificate via Let's Encrypt
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

---

## 4. Environment Variables & Configuration

The application works out of the box with default paths. You can optionally set:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `5000` | Port for Flask/Gunicorn web server |
| `FLASK_ENV` | `production` | Set to `development` for local debugging |
| `DATA_DIR` | `Dataset` | Path to directory containing GFW Excel files |

---

## 5. Verifying the Deployment

Run health checks against the running service:
```bash
# Check web page:
curl -I http://localhost:5000/

# Check API health:
curl -s http://localhost:5000/api/overview | grep "total_loss_ha"
```
If output contains `1933252.0`, the deployment is successful and serving accurate data!
