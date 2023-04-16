from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,predictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('homepage.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('formpage.html')
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            table=float(request.form.get('table')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        data_df=data.get_data_as_df()
        pred_pipeline=predictPipeline()
        pred_val=pred_pipeline.predict(data_df)
        pred_val=round(pred_val[0],2)
        return render_template('resultpage.html',final_result=pred_val)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)

