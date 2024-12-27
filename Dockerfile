#Use a lightweight Python image
FROM python:3.11-slim

#Set the working directory in the container
WORKDIR /app

#Copy the requirements file
COPY requirements.txt /app/requirements.txt

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy the entire app into the container
COPY . /app

#Expose the port Flask runs on (default: 5000)
EXPOSE 5000

#Set the default command to run your Flask app
CMD ["python", "flask_app.py"]