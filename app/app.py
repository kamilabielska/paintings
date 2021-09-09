import os
import pickle
import numpy as np
import torch
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go
import requests

from dash.dependencies import Input, Output, State
from torchvision import transforms
from PIL import Image
from io import BytesIO
from model_class import ResModel

app = dash.Dash(__name__, update_title=None)
app.title = 'who painted that?'
server = app.server


## PREPARE THE DATA AND THE MODEL
nclasses = 12

filename = {12:'model_resnet50_12_8956.pth',
            15:'model_resnet50_15_9212.pth',
            50:'model_resnet50_8178.pth'}
    
test_images = pickle.load(open(os.path.join('arrays','test_images_'+str(nclasses)),'rb'))
test_labels = pickle.load(open(os.path.join('arrays','test_labels_'+str(nclasses)),'rb'))
which = pickle.load(open(os.path.join('arrays','which_'+str(nclasses)),'rb'))
        
unique = sorted(list(set(test_labels)))
options = [{'label': x, 'value': x} for x in unique]

n_instances = len(test_images)

model = ResModel(nclasses)
model.load_state_dict(torch.load(filename[nclasses], map_location=torch.device('cpu')))
model.eval()
        
transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
    
def display_image(image_raw, width=590, height=525):
    fig = go.Figure()

    img_width, img_height = image_raw.size
    scale_factor = min([width/img_width, height/img_height])
    scaled_width, scaled_height = img_width*scale_factor, img_height*scale_factor

    fig.add_trace(
        go.Scatter(
            x=[0, scaled_width],
            y=[0, scaled_height],
            mode='markers',
            marker_opacity=0
        )
    )

    fig.update_xaxes(
        visible=False,
        range=[0, scaled_width]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, scaled_height],
        scaleanchor='x'
    )

    fig.add_layout_image(
        dict(
            x=0,
            sizex=scaled_width,
            y=scaled_height,
            sizey=scaled_height,
            xref='x',
            yref='y',
            opacity=1,
            layer='below',
            source=image_raw)
    )

    fig.update_layout(
        width=scaled_width,
        height=scaled_height,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    
    return fig

blank_image = display_image(Image.new('RGB', (50, 50), (255, 255, 255)))


## SCOREBOARD
def create_score_card(title, id_text):

        card = dbc.Card([
            dbc.CardHeader(
                title,
                style={
                    'textAlign': 'center',
                    'padding': '4px 0px 4px 0px'
                }
            ),
            dbc.CardBody([
                html.P(
                    '0',
                    id=id_text,
                    style={
                        'textAlign': 'center',
                        'fontSize': '28px',
                        'paddingBottom': '0px',
                        'marginBottom': '0px'
                    }
                )
            ],
                style={'padding': '0px 0px 0px 0px'}
            )
        ])
        
        return card

you = create_score_card('you', 'your-points')
network = create_score_card('neural network', 'network-points')

scoreboard_container = dbc.Card([
        html.Div([
            html.H3(
                'scoreboard',
                style={'textAlign': 'center'}
            ),
            dbc.Row([
                dbc.Col(you, width=4, style={'paddingRight': '2px'}),
                dbc.Col(network, width=4, style={'paddingLeft': '2px'})
            ],
                justify='center',
                align='stretch'
            )
        ], 
            style={
                'display': 'inline-block',
                'marginTop':'auto',
                'marginBottom':'auto'
            }
        )
], 
    body=True,
    style={
        'height': '35%',
        'backgroundColor': '#E8E8E8'
    }
)


## MAIN PANEL
summary_text = "**your choice**: {}\n**network's choice**: {}\n**author of the artwork**: {}"

main_container = dbc.Card([
    html.H3(
        'who painted that?',
        style={
            'textAlign': 'left',
            'marginBottom': '20px'
        }
    ),
    dcc.Dropdown(
        options=options,
        id='dropdown',
        placeholder='select an artist',
        style={'marginBottom': '5px'}
    ),
    dbc.Button(
        'confirm',
        id='confirm-button',
        color='primary',
        block=True,
        style={'marginBottom': '25px'}
    ),
    dcc.Markdown(
        summary_text.format('','',''),
        id='summary',
        style={'whiteSpace': 'pre-wrap'}
    ),
    dbc.Button(
        'next artwork',
        id='next-button',
        color='primary',
        style={
            'marginTop': '15px',
            'display': 'inline-block',
            'width': '35%',
            'margin':'auto'
        }
    ),
    html.P(
        '0/'+str(n_instances),
        id='counter',
        style={
            'paddingTop': '5px',
            'paddingBottom': '15px',
            'display': 'inline-block',
            'margin':'auto'
        }
    )
],
    body=True,
    style={
        'marginBottom': '5px',
        'paddingLeft': '7%',
        'paddingRight': '7%',
        'height': '65%',
        'backgroundColor': '#E8E8E8'
    }
)


## WELCOME AND THE END CARDS
welcome_card = dbc.Modal([
    dbc.ModalHeader('Welcome'),
    dbc.ModalBody(
        "Try your luck (or knowledge) against a neural network and see who is "+
        "better at guessing the authors of presented artworks. Choose artist's "+
        "name from the list and press 'confirm'. Every right guess is worth 1 point."
    ),
    dbc.ModalFooter(
        dbc.Button(
            "Let's go!",
            id='close-button',
            className='ml-auto',
            n_clicks=0,
            style={
                'display': 'inline-block',
                'margin': 'auto'
            }
        )
    ),
], 
    id='welcome-card',
    is_open=True,
    centered=True
)

the_end = dbc.Modal([
    dbc.ModalHeader('The end'),
    dbc.ModalBody(
        'You made it to the end, congrats!',
        id='end-message'
    ),
    dbc.ModalFooter(
        dbc.Button(
            'close',
            id='end-button',
            className='ml-auto',
            n_clicks=0,
            style={
                'display': 'inline-block',
                'margin': 'auto'
            }
        )
    ),
],
    id='the-end-card',
    is_open=False,
    centered=True
)


## LAYOUT
app.layout = html.Div([
    dbc.Row([
        welcome_card,
        dbc.Col([
            dcc.Graph(
                figure=blank_image,
                id='image',
                config={'doubleClick': 'autosize'},
                style={'width': '100%'}
            )
        ], width=6),
        dbc.Col([
            main_container,
            scoreboard_container,
        ], width=5),
        the_end
    ], 
        justify='center'
    ),
    dcc.Location(id='url'),
    html.Div(
        children='1366 657',
        id='window-size',
        hidden=True,
        style={'display':'none'}
    )
], 
    style={'margin': '3% 7% 3% 7%'}
)


## CALLBACKS 
app.clientside_callback(
    """
    function(href) {
        var w = window.innerWidth;
        var h = window.innerHeight;
        const str = String(w) + ' ' + String(h);
        return str;
    }
    """,
    Output('window-size', 'children'),
    Input('url', 'href')
)

@app.callback(
    Output('welcome-card', 'is_open'),
    Input('close-button', 'n_clicks')
)
def close_welcome_card(n_clicks):
    if n_clicks:
        return False
    else:
        return True

@app.callback(
    Output('summary', 'children'),
    Output('your-points', 'children'),
    Output('network-points', 'children'),
    Input('confirm-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('dropdown', 'value'),
    State('your-points', 'children'),
    State('network-points', 'children'),
    State('counter', 'children'),
    prevent_initial_call=True
)
def confirm_choice(confirm_button, next_button, drop_input, ypoints, npoints, counter):
    button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'confirm-button': 
        n = int(counter.split('/')[0]) - 1
        real = test_labels[which[n]]

        with torch.no_grad():
            response = requests.get(test_images[which[n]])
            image_raw = Image.open(BytesIO(response.content)).convert('RGB')
            image = transformation(image_raw)
            pred = np.argmax(model(image.view(1, 3, 224, 224)))

        your_points = str(int(ypoints)+1) if drop_input == real else ypoints
        network_points = str(int(npoints)+1) if unique[pred] == real else npoints

        return summary_text.format(drop_input, unique[pred], real), your_points, network_points
    
    elif button_id == 'next-button':
        return summary_text.format('', '', ''), ypoints, npoints

@app.callback(
    Output('image', 'figure'),
    Output('counter', 'children'),
    Output('dropdown', 'value'),
    Output('the-end-card', 'is_open'),
    Output('next-button', 'disabled'),
    Output('end-message', 'children'),
    Input('next-button', 'n_clicks'),
    Input('end-button', 'n_clicks'),
    State('counter', 'children'),
    State('your-points', 'children'),
    State('network-points', 'children'),
    State('end-message', 'children'),
    State('window-size', 'children')
)
def load_next_image(n_clicks1, n_clicks2, counter, ypoints, npoints, message, window):
    n = int(counter.split('/')[0])
    w, h = window.split(' ')
    print(w, h)
    width, height = int(w)/2 - 0.08*int(w), int(h) - 0.2*int(h)
    
    if n >= n_instances:
        button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        response = requests.get(test_images[which[n-1]])
        image_raw = Image.open(BytesIO(response.content)).convert('RGB')
        fig = display_image(image_raw, height=height, width=width)
        card_state = False if button_id == 'end-button' else True
        
        if int(ypoints) > int(npoints):
            end_message = 'You win! Congrats, you are better than the neural network.'
        elif int(ypoints) == int(npoints):
            end_message = 'Tie! Congrats, you are as good as the neural network.'
        else:
            end_message = 'You lose. Unfortunately, the neural network was better this time.'
            
        return fig, '/'.join([str(n), str(n_instances)]), None, card_state, True, end_message
        
    else:
        response = requests.get(test_images[which[n]])
        image_raw = Image.open(BytesIO(response.content)).convert('RGB')
        fig = display_image(image_raw, height=height, width=width)
        return fig, '/'.join([str(n+1), str(n_instances)]), None, False, False, message

@app.callback(
    Output('confirm-button', 'disabled'),
    Input('next-button', 'n_clicks'),
    Input('confirm-button', 'n_clicks'),
    State('counter', 'children'),
    prevent_initial_call=True
)
def disable_confirm_button(next_button, confirm_button, counter):
    button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    n = int(counter.split('/')[0])
    
    if button_id == 'next-button':
        if n >= n_instances:
            return True
        else:
            return False
    elif button_id == 'confirm-button':
        return True
    else:
        raise dash.exceptions.PreventUpdate
        

if __name__ == '__main__':
    app.run_server(debug=True)