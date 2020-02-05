import smtplib

def sendmessage(ticker, prediction):
    username = "stockpredictorpython@gmail.com"
    password = "stockpredictor"
    atext = "7036758188@sms.myboostmobile.com"
    ntext = "5105226389@txt.att.net"

    message = "The projected value of " + ticker + " stock tomorrow is " + str(prediction) + " dollars"

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(username, password)
    server.sendmail(username, atext, message)
    server.quit()