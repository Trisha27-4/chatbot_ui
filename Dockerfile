# Use slim Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy everything from chatbot_ui into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Expose Flask port
EXPOSE 5000

# Set Flask env variables
ENV FLASK_APP=app_openai.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run Flask app
CMD ["flask", "run"]
