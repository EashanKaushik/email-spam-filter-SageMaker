import json
import boto3
import os
import email
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

VOCAB = 9013
try:
    ENDPOINT = os.environ["ENDPOINT_NAME"]
except Exception as ex:
    ENDPOINT = None
    
REPLY_EMAIL = os.environ["REPLY_EMAIL"]
    
def lambda_handler(event, context):
    # print(event)
    
    
    s3 = boto3.client("s3")
    if not ENDPOINT:
        sagemaker = boto3.client("sagemaker")
        response = sagemaker.list_endpoints()
        EndpointName = response["Endpoints"][0]["EndpointName"]
    else:
        EndpointName = ENDPOINT
    # print(EndpointName)
    
    # return
    sagemaker_runtime = boto3.client("sagemaker-runtime", "us-east-1")
    
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
    
        data = s3.get_object(Bucket=bucket, Key=key)
        contents = data['Body'].read()
        email_msg = email.message_from_bytes(contents)
        # receiver_email_address = email_msg["To"]
        receiver_email_address = REPLY_EMAIL
        
        recipient_email_address = email_msg['From']
        receive_date = email_msg['date']
        subject = email_msg['subject']
        
        
        
        if email_msg.is_multipart():
            for part in email_msg.get_payload():
                if part.get_content_type() == 'text/plain':
                    body = part.get_payload()
        else:
            body = email_msg.get_payload()
            
        encoded_message = body_preprocessing(body)
        prediction = sagemaker_runtime.invoke_endpoint(EndpointName=EndpointName, ContentType='application/json',Body=json.dumps(encoded_message.tolist()))
        print("prediction", prediction)
        response = json.loads(prediction["Body"].read().decode("utf-8"))
        spam = int(response.get('predicted_label')[0][0])
        score = str(float(response.get('predicted_probability')[0][0]) * 100)
        
        print(spam, score)
        
        
        email_spam_filter(recipient_email_address, receive_date, subject, body.strip(), spam, score, receiver_email_address)
        
    return {
        'statusCode': 200,
        'body': json.dumps('Success')
    }

def body_preprocessing(body):
    body = [body.strip()]
    
    one_hot_test_messages = one_hot_encode(body, VOCAB)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, VOCAB)
    
    return encoded_test_messages

def email_spam_filter(recipient_email_address, receive_date, subject, body, spam, score, receiver_email_address):
    ses = boto3.client('ses')
    
    SUBJECT = "SPAM Identification"
    
    BODY_TEXT = ("")
    
    BODY_HTML = """<html>
                    <head></head>
                    <body>
                      <p>
                        """+"We received your email sent at {} with the subject {}. \
                        Here is a 240 character sample of the email body: {}. \
                        The email was categorized as {} with a {}% confidence"\
                        .format(receive_date, subject, body, "SPAM" if spam == 1 else "NOT SPAM", score)+"""
                      </p>
                    </body>
                    </html>
                """
    
    CHARSET = "UTF-8"
    
    message = "We received your email sent at {} with the subject <i>{}<i/>. Here is a 240 character sample of the email body:\
    <b>{}</b>. The email was categorized as {} with a {}% confidence".format(receive_date, subject, body, "SPAM" if spam == 1 else "NOT SPAM", score)
    
    response = ses.send_email(
        Destination={
            'ToAddresses': [
                recipient_email_address
            ],
        },
        Message={
            'Body': {
                'Html': {
                    'Charset': CHARSET,
                    'Data': BODY_HTML,
                },
                'Text': {
                    'Charset': CHARSET,
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': SUBJECT,
            },
        },
        Source=receiver_email_address)
    print(response)