FROM python:3.11

WORKDIR /code

# Install system dependencies required for PyTorch and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./ /code/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000" , "app:app"]
