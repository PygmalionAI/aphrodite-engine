import gradio as gr
import requests
import argparse

parser = argparse.ArgumentParser(
	prog='Parallel Test',
	description='Test'
)

parser.add_argument('-r', '--rows', type=int, default=20, required=False)
parser.add_argument('-dr', '--defaultrows', type=int, default=1, required=False)
args = parser.parse_args()

maxrows = args.rows
defaultrows = args.defaultrows
currentrows = defaultrows

async def doRequest(input, n):
    print(f'Generating {n} response(s).')
    print(f'Current rows {n}')
    body = {
    "model": "goliath",
    "prompt": [input],
    "temperature": 1.2,
    "min_p" : 0.05,
    "stream" : False,
    "n" : n,
    "best_of" : n,
    }
    print('Getting reply')
    reply = requests.post(url='https://waifu.pygmalion.chat/v1/completions',
        json=body,
        headers={'X-API-KEY':'pyg-goliath-api-lol', 'Content-Type': 'application/json', 'accept': 'application/json'}
    )
    output = [ n['text'] for n in reply.json()['choices'] ] 
    # some arrays of text
    for i in range(maxrows - n):
        output.append(f'Output {n}')
    return output


with gr.Blocks() as demo:
    slider = gr.Slider(minimum=1, maximum=maxrows, label="Outputs", step=1, value=defaultrows)
    with gr.Row():
        with gr.Column():
            input = gr.Textbox()
            submit = gr.Button(value='Submit')
        with gr.Column():
             outputs = [gr.Textbox(interactive=False,visible=i < defaultrows, label=f"Output {i + 1}") for i in range(maxrows)]
    
    def updateFields(x):
        global currentrows 
        currentrows = x
        fields = [gr.update(visible = (n < x) ) for n in range(maxrows)] 
        return fields
        
    def getoutputs():
        global currentrows
        return outputs[:currentrows]
    
    slider.change(fn=updateFields, inputs=[slider], outputs=outputs)
    submit.click(fn=doRequest, inputs=[input, slider], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
    
