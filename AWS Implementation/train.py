import json
import boto3
import joblib
import os
from io import BytesIO

# --- CONFIGURATION ---
BUCKET_NAME = 'breast-cancer-prediction-models'
MODEL_KEY = 'latest_model.pkl'
DYNAMODB_TABLE = 'Patient_Entries'  # Updated to match your existing table
SNS_TOPIC_ARN = 'arn:aws:sns:eu-central-1:469541406278:MalignantAlerts'

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
table = dynamodb.Table(DYNAMODB_TABLE)

def load_model_from_s3():
    print("Loading model from S3...")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    model_stream = BytesIO(response['Body'].read())
    model = joblib.load(model_stream)
    print("Model loaded successfully.")
    return model

# Load model globally to reuse across invocations
model = load_model_from_s3()

def process_doctor_feedback(body):
    """Handles the Tick/Cross feedback from the doctor."""
    case_id = body.get('id')
    resolution = body.get('resolution')  # 'confirmed_malignant' or 'confirmed_benign'

    if not case_id or not resolution:
        return {'statusCode': 400, 'body': json.dumps('Missing id or resolution')}

    print(f"Processing feedback for Case {case_id}: {resolution}")

    # Update DynamoDB
    try:
        table.update_item(
            Key={'id': case_id},
            UpdateExpression="SET doctor_resolution = :r, status = :s",
            ExpressionAttributeValues={
                ':r': resolution,
                ':s': 'Resolved'
            }
        )
        return {'statusCode': 200, 'body': json.dumps(f"Case {case_id} resolved as {resolution}")}
    except Exception as e:
        print(f"Error updating DynamoDB: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps(f"Database error: {str(e)}")}

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        
        # Parse body
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event

        # CHECK OPERATION TYPE
        if body.get('operation') == 'feedback':
            return process_doctor_feedback(body)

        # --- PREDICTION LOGIC ---
        # Expecting 'features' list in the body
        features = body.get('features')
        case_id = body.get('id', 'unknown_id')

        if not features:
            return {
                'statusCode': 400,
                'body': json.dumps("Error: 'features' list is required.")
            }

        # Predict directly from list (No Pandas needed)
        # Scikit-learn expects a 2D array: [[f1, f2, ...]]
        prediction = model.predict([features])
        result = 'M' if prediction[0] == 1 else 'B'

        print(f"Prediction for {case_id}: {result}")

        # Save to DynamoDB
        item = {
            'id': case_id,
            'features': str(features),  # Store as string or list
            'prediction': result,
            'status': 'Pending Review'
        }
        table.put_item(Item=item)

        # Send SNS Alert if Malignant
        if result == 'M':
            message = (
                f"URGENT: Malignant Case Detected\n\n"
                f"Case ID: {case_id}\n"
                f"Prediction: Malignant (M)\n"
                f"Status: Pending Doctor Review\n\n"
                f"Please log in to the dashboard to review this case immediately."
            )
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject=f"Alert: Malignant Case {case_id}"
            )

        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': result, 'id': case_id})
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Internal Server Error: {str(e)}")
        }