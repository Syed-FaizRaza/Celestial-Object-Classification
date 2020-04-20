from flask import Flask,render_template,request,url_for,redirect,Blueprint,flash
from flask_table import Table, Col
import cv2
from keras.models import load_model
import os
import numpy as np
from keras import backend as K
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

gallery = Blueprint('gallery', __name__, template_folder='templates', static_folder='static')
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + "seconds")

    else:
        print ("Toc: start time not set")

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def sixteen_pieces(data):

    norm = ((data).astype(np.float32))
    norm_tl = norm[:45, :45, :]  # top-left
    norm_tr = norm[:45, 24:, :]  # top-right
    norm_bl = norm[24:, :45, :]  # bottom-left
    norm_br = norm[24:, 24:, :]  # bottom-right

    norm_flip = np.fliplr(norm)
    norm_flip_tl = norm_flip[:45, :45, :]
    norm_flip_tr = norm_flip[:45, 24:, :]
    norm_flip_bl = norm_flip[24:, :45, :]
    norm_flip_br = norm_flip[24:, 24:, :]

    rot45 = rotateImage((data.astype(np.float32)), 45)
    rot_tl = rot45[:45, :45, :]
    rot_tr = rot45[:45, 24:, :]
    rot_bl = rot45[24:, :45, :]
    rot_br = rot45[24:, 24:, :]

    rot45_flip = np.fliplr(rot45)
    rot_flip_tl = rot45_flip[:45, :45, :]
    rot_flip_tr = rot45_flip[:45, 24:, :]
    rot_flip_bl = rot45_flip[24:, :45, :]
    rot_flip_br = rot45_flip[24:, 24:, :]

    return ([norm_tl, norm_tr, norm_bl, norm_br,
             norm_flip_tl, norm_flip_tr, norm_flip_bl, norm_flip_br,
             rot_tl, rot_tr, rot_bl, rot_br,
             rot_flip_tl, rot_flip_tr, rot_flip_bl, rot_flip_br])

#_____________________________________________________________________________________________________________________________________________________________________

def pred_16(imgg0):
    tic()
    path_model = 'model and weights/'
    model1_2 = load_model(path_model+'model1_45_45_3_cnn_ellip_spiral_1.1_1.2_1.3.h5')

    model2_71_72_73_2 = load_model(path_model+'model2_45_45_3_cnn_ellip_roundness.h5')

    model5_21_22_2 = load_model(path_model+'model5_45_45_3_cnn_spiral_edge_on_not_edgeon.h5')  # model5
    model6_91_92_2 = load_model(path_model+'model6_45_45_3_cnn_spiral_buldge_of_edge_on_9.1_9.3.h5')
    model7_31_32_2 = load_model(path_model+'model7_45_45_3_cnn_spiral_Is-Therer_a_bar_on_not_edge_on(Ques2).h5')
    model8_52_53_2 = load_model(path_model+'model8_45_45_3_cnn_spiral_not_edge_on_Buldge_5.2_5.3.h5')
    toc()
    row = 69
    col = 69

    spiral=0

    edge_on=0
    less_buldge=0
    prominent_buldge=0

    not_edge_on=0
    bar=0
    not_bar=0
    less_buld=0
    prominent_buld=0



    elliptical=0
    face_on=0 #E0
    E5=0
    with_some_angle=0 #e7

    others=0
    model1 = model1_2

    model2_71_72_73= model2_71_72_73_2

    imgg1 = cv2.resize(imgg0, (row, col))
    imgg2 = sixteen_pieces(imgg1)

    for i,j in enumerate(imgg2):
        imgg2[i] = np.expand_dims(imgg2[i], axis=0)
        y_pred =np.array(model1.predict(imgg2[i])).T # 3,1
        if(np.amax(y_pred)>0.7000):

            if (y_pred[0,0] > y_pred[1,0]) and (y_pred[0,0] > y_pred[2,0]):
                elliptical = elliptical + 1
                y_pred2 = np.array(model2_71_72_73.predict(imgg2[i])).T  # 3,1
                if (y_pred2[0, 0] > y_pred2[1, 0] and y_pred2[0, 0] > y_pred2[2, 0]):
                    face_on = face_on + 1
                elif (y_pred2[1, 0] > y_pred2[0, 0] and y_pred2[1, 0] > y_pred2[2, 0]):
                    with_some_angle = with_some_angle + 1
                else:
                    E5 = E5 + 1


            elif (y_pred[1,0] > y_pred[0,0]) and (y_pred[1,0] > y_pred[2,0]):
                model5_21_22 = model5_21_22_2
                model6_91_92 = model6_91_92_2
                model7_31_32 = model7_31_32_2
                model8_52_53 = model8_52_53_2

                spiral = spiral + 1
                y_pred3 = np.array(model5_21_22.predict(imgg2[i])).T  # 2,1
                #predict edge-on(2.1),not edge-on (2.2)

                if (y_pred3[1, 0] >= y_pred3[0, 0]):#edge-on
                        not_edge_on = not_edge_on + 1
                        y_pred_31_32 = np.array(model7_31_32.predict(imgg2[i])).T  # 2,1
                        #predict bar(3.1), not-bar (3.2)

                        y_pred_52_53 = np.array(model8_52_53.predict(imgg2[i])).T #2,1 output
                        #predict less buldge of not edge-on spiral galaxies(5.2,less),prominent bulde (5.3,prominent)
                        #we have to calculate buldge of both bar and not-bar galaxies

                        if (y_pred_31_32[1, 0] > y_pred_31_32[0, 0]):  # bar-not bar
                            not_bar = not_bar + 1
                            if (y_pred_52_53[0, 0] > y_pred_52_53[1, 0]):  # lessbuldge of not edge-on not bar
                                less_buld = less_buld + 1
                            else:  # lessbuldge of not edge-on
                                prominent_buld = prominent_buld + 1


                        else:  # bar-not bar
                            bar = bar + 1
                            if (y_pred_52_53[0, 0] > y_pred_52_53[1, 0]):  # less buldge of not edge-on with bar
                                less_buld = less_buld + 1
                            else:  # lessbuldge of not edge-on
                                prominent_buld = prominent_buld + 1
                else: #edge-on
                    edge_on = edge_on + 1
                    y_pred_91_92 = np.array(model6_91_92.predict(imgg2[i])).T  # 2,1
                    if (y_pred_91_92[0, 0] > y_pred_91_92[1, 0]):
                        less_buldge = less_buldge + 1
                    else:
                        prominent_buldge = prominent_buldge + 1


            elif (y_pred[2,0] > y_pred[0,0]) and (y_pred[2,0] > y_pred[1,0]):
                others=others+1



    if(elliptical > spiral and elliptical > others):

            # elliptical = 0,face_on = 0,E5 = 0,with_some_angle = 0
            return (1,face_on,
                    with_some_angle,
                    E5
                    )
    elif(spiral > elliptical and spiral > others):

            return(2,
               edge_on,not_edge_on,
               less_buldge,prominent_buldge,
               bar,not_bar,
               less_buld,prominent_buld
               )#parameter needs to address if the galaxy is spiral
    else:
            return(3,4,5)
#_____________________________________________________________________________________________________________________________________________________________________
import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


### Function to send the email ###
def send_an_email(image,body):

    toaddr = 'faizraza134@gmail.com'
    me = 'fyp376166@gmail.com'
    subject = "Report"

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = toaddr
    msg.preamble = "test "
    body = MIMEText(body)
    msg.attach(body)
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open("static/"+image, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="your Input Image.jpg"')
    msg.attach(part)

    try:
       s = smtplib.SMTP('smtp.gmail.com', 587)
       s.ehlo()
       s.starttls()
       s.ehlo()
       s.login(user = 'fyp376166@gmail.com', password = 'ABCDefg123')
       #s.send_message(msg)
       s.sendmail(me, toaddr, msg.as_string())
       s.quit()
    #except:
    #   print ("Error: unable to send email")
    except smtplib.SMTPException as error:
          print ("Error")

newVar=None
img=[]
# global counter
hists = os.listdir('static/plots')
hists = ['plots/' + file for file in hists]
@app.route("/")
def index():
    new_desc = []
    global newVar
    if newVar == None:
        newVar=1
        return render_template("google.html")
    else:
        return render_template("index.html",hists=hists,new_desc=new_desc)
@app.route("/upload", methods=['POST'])

def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        filename = file.filename
        dest = "/".join([target,filename])
        img.append(dest)
        file.save(dest)
    if img != []:
        imgg = (cv2.imread(img[-1]))

        try:

            K.clear_session()
            imgg0 = imgg / 255
            rett = np.array(pred_16(imgg0.astype(np.float32)))
            if rett[0] == 1:  # elliptical
                desc = os.listdir('static/desc')
                new_desc=[]
                for i in desc:
                    if i == 'eliptical.png':
                        new_desc.append('desc/' + i)
                if rett[1] > rett[2] and rett[1] > rett[3]:
                    string1 = "Elliptical Galaxy"
                    string2 = "This galaxy is might be E0"
                elif rett[2] > rett[1] and rett[2] > rett[3]:
                    string1 = "Elliptical Galaxy"
                    string2 = "This galaxy is might be E5"
                elif rett[3] > rett[2] and rett[3] > rett[1]:
                    string1 = "Elliptical Galaxy"
                    string2 = "This galaxy is might be E7"
                # return render_template("complete.html", image_name=filename,string1=string1,string2=string2,new_desc=new_desc)
                send_an_email(filename,string1+string2)
                return render_template('index.html',image_name=filename,string1=string1,string2=string2, hists=hists,new_desc=new_desc)
            elif rett[0] == 2:
                desc = os.listdir('static/desc')
                new_desc=[]
                for i in desc:
                    if i == 'Edge with bulge.png':
                        new_desc.append('desc/' + i)
                if (rett[1] > rett[2]):#if edge-on spiral then do this
                    if (rett[3]>=rett[4]) :
                        string1 = "This is Spiral Galaxy"
                        string2 = "This galaxy might have edges and bulge (type 1)"
                    elif (rett[4]>rett[3]):
                        string1 = "This is Spiral Galaxy"
                        string2 = "This galaxy might have edges but no bulge(type 2)"
                    send_an_email(filename, string1 + string2)
                    return render_template('index.html',image_name=filename,string1=string1,string2=string2,hists=hists,new_desc=new_desc)
                elif (rett[1] < rett[2]):
                    desc = os.listdir('static/desc')
                    new_desc = []
                    for i in desc:
                        if i == 'bar and bulge1.png':
                            new_desc.append('desc/' + i)
                    if (rett[5]>rett[6]) and (rett[7]>rett[8]) :
                        string1 = "This is Spiral Galaxy without Edge "
                        string2 = "This galaxy might be with bar and less prominent bulge (type 1)"
                    elif (rett[5]>rett[6]) and (rett[7]<rett[8]) :
                        string1 = "This is Spiral Galaxy without Edge "
                        string2 = "This galaxy might be with bar and more prominent bulge (type 2)"
                    elif (rett[5]<rett[6]) and (rett[7]>rett[8]) :
                        string1 = "This is Spiral Galaxy without Edge "
                        string2 = "This galaxy might be without bar and less prominent bulge (type 3)"
                    else:
                        string1 = "This is Spiral Galaxy without edge "
                        string2 = "This galaxy might be without a bar and more prominent bulge (type 4)"
                        send_an_email(filename, string1 + string2)
                    return render_template('index.html',image_name=filename,string1=string1,string2=string2, hists=hists,new_desc=new_desc)
                # return (2,edge_on, not_edge_on,less_buldge, prominent_buldge,bar, not_bar,less_buld, prominent_buld)  #

            elif rett[0] == 3:
                K.clear_session()
                desc = os.listdir('static/desc')
                new_desc = []
                for i in desc:
                    if i == 'Irregular.png':
                        new_desc.append('desc/' + i)
                string1='this is irregular galaxy'
                string2='Might be of any of  any shape'
                send_an_email(filename, string1 +" "+string2)
                return render_template('index.html',image_name=filename,string1=string1,string2=string2, hists=hists,new_desc=new_desc)
        except IndexError as error:
            print(error)
    else:
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run('localhost',port=8080, debug=True)





