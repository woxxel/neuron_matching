import os,pickle,h5py,cv2#from matching import *
import numpy as np
import scipy as sp

from plotly import graph_objects as go, express as px
from plotly.subplots import make_subplots

from matching.utils import build_remap_from_shift_and_flow

import dash
from dash import dcc,html
from dash.dependencies import Input, Output

mousePath = 'data/556wt'

filePath = os.path.join(mousePath,'matching/neuron_registration_.pkl')
with open(filePath,'rb') as f_open:
    ld = pickle.load(f_open)
    assignments = ld['results']['assignments']
    sessionData = ld['data']
# print(sessionD.keys())
paths = [os.path.join(mousePath,sessionPath,'OnACID_results.hdf5') for sessionPath in os.listdir(mousePath) if 'Session' in sessionPath]
paths.sort()
dims = (512,512)

X = np.arange(0, dims[0])
Y = np.arange(0, dims[1])
X, Y = np.meshgrid(X, Y)

use_opt_flow=True

app = dash.Dash(__name__)
fig=go.Figure()

c=0

A = {}
for s,path in enumerate(paths):
    if s>10: break
    if os.path.exists(path):
        # print(s,path)
        file = h5py.File(path,'r')
        # only load a single variable: A

        data =  file['/A/data'][...]
        indices = file['/A/indices'][...]
        indptr = file['/A/indptr'][...]
        shape = file['/A/shape'][...]
        A[s] = sp.sparse.csc_matrix((data[:], indices[:],
            indptr[:]), shape[:])
        
        # if s>0:
        #     x_remap,y_remap = build_remap_from_shift_and_flow(dims,sessionData[s]['remap']['shift'],sessionData[s]['remap']['flow'] if use_opt_flow else None)
            
        #     A[s] = sp.sparse.hstack([
        #         sp.sparse.csc_matrix(                    # cast results to sparse type
        #             cv2.remap(
        #                 fp.reshape(dims),   # reshape image to original dimensions
        #                 x_remap, y_remap,                 # apply reverse identified shift and flow
        #                 cv2.INTER_CUBIC                 
        #             ).reshape(-1,1)                       # reshape back to allow sparse storage
        #             ) for fp in footprints.toarray().T        # loop through all footprints
        #         ])
        # else:
        #     A[s] = footprints

def plot_fp(fig,c):
            
    for s,path in enumerate(paths):
        # if os.path.exists(path):
        # print(s,path)
        if s > 10: break

        idx = assignments[c,s]
        # print('footprint:',s,idx)
        if np.isfinite(idx):
            # file = h5py.File(path,'r')
            # # only load a single variable: A

            # data =  file['/A/data'][...]
            # indices = file['/A/indices'][...]
            # indptr = file['/A/indptr'][...]
            # shape = file['/A/shape'][...]
            # A = sp.sparse.csc_matrix((data[:], indices[:],
            #     indptr[:]), shape[:])

            A_fp = A[s][:,int(idx)].reshape(dims).todense()

            if s>0:
                ## use shift and flow to align footprints - apply reverse mapping
                x_remap,y_remap = build_remap_from_shift_and_flow(dims,sessionData[s]['remap']['shift'],sessionData[s]['remap']['flow'] if use_opt_flow else None)
                A_fp = cv2.remap(A_fp,
                    x_remap, y_remap, # apply reverse identified shift and flow
                    cv2.INTER_CUBIC                 
                )
            
            A_fp /= A_fp.max()
            A_fp[A_fp<0.1*A_fp.max()] = np.NaN

            fig.add_trace(go.Surface(x=X,y=Y,z=A_fp+s,showscale=False,name=f'Session {s}',opacity=0.5))                        
            
plot_fp(fig,c)
nC = assignments.shape[0]
# print(fig)
app.layout = html.Div([
    dcc.Graph(
        id='footprints',
        figure=fig
    ),
    dcc.Slider(
        id='cluster',
        min=0,
        max=nC,
        step=None,
        value=0,
        # marks=dict(zip(np.arange(0,nC,50),np.arange(0,nC,50).astype(str)))
    ),
    html.Div(id='slider-output')
    ]
)
#    Output('slider-output','children'),
@app.callback(
    [Output('footprints','figure'),
    Output('slider-output','children')
    ],
    [Input('cluster','value')]
)
def update_output(value):
    fig=go.Figure()
    plot_fp(fig,value)
    return fig, f"Now displaying neuron {value}"
#    return f'You have selected value {value}'
# if __name__ == '__main__':
app.run_server(debug=True)

# match.plot_footprints(0,use_plotly=True)

