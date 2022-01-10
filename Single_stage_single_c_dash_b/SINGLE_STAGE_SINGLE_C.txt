
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import dash_table
from dash.exceptions import PreventUpdate
from scipy.optimize import minimize, linprog
import base64
import io


class TransPortationModel:
    def __init__(self):
        self.no_of_f = 1
        self.no_of_c = 1
        self.scale = 1
        self.facility_names = []
        self.customer_names = []
        self.index_headers = []

    def make_me_none(self):
        size = self.no_of_f * self.no_of_c
        temp = [np.nan] * size
        self.cost_for_f_to_cust = np.array(temp)
        self.f_capacity_constraint = np.array(temp)
        self.c_demand_constraint = np.array(temp)

    def create_data_frames(self):
        size = self.no_of_f * self.no_of_c
        no_of_f = self.no_of_f
        no_of_c = self.no_of_c
        self.data_frame_index = []
        self.data_frame_index.append(self.facility_names)
        self.data_frame_index.append(self.customer_names)
        cost = self.cost_for_f_to_cust[:size].reshape((no_of_f, no_of_c))
        capa = self.f_capacity_constraint[:no_of_f].reshape((1,no_of_f))
        demand = self.c_demand_constraint[:no_of_c].reshape((1,no_of_c))
        self.df_cost = pd.DataFrame(cost, self.data_frame_index[0], self.data_frame_index[1])
        df=pd.DataFrame(self.data_frame_index[0],self.data_frame_index[0],[self.index_headers[0]])
        self.df_cost = pd.concat([df, self.df_cost], axis=1)
        self.df_demand = pd.DataFrame(demand, ["Demand"],self.data_frame_index[1],)
        self.df_capacity = pd.DataFrame(capa, ["Capacity"],self.data_frame_index[0])

    def solve_problem(self):
        no_of_f = self.no_of_f
        no_of_c = self.no_of_c
        no_of_decision_variable = no_of_c * no_of_f

        A_ub = np.zeros((no_of_f, no_of_c))
        A_ub[0] = 1
        A_ub = A_ub.reshape(no_of_decision_variable)
        for i in range(1, no_of_f):
            z = np.zeros((no_of_f, no_of_c))
            z[i] = 1
            z = z.reshape(no_of_decision_variable)
            A_ub = np.vstack((A_ub, z))
        for i in range(0, no_of_c):
            z = np.zeros((no_of_f, no_of_c))
            z[0:, i] = -1
            z = z.reshape(no_of_decision_variable)
            A_ub = np.vstack((A_ub, z))

        demand_constraint_neg = -1 * self.c_demand_constraint
        b_ub = np.append(self.f_capacity_constraint, demand_constraint_neg)

        cost_function = self.cost_for_f_to_cust.copy()
        Max_prod = [None] * no_of_decision_variable
        Min_prod = np.zeros(no_of_decision_variable)
        bounds = tuple(zip(Min_prod, Max_prod))

        Min_cf = linprog(cost_function, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method="interior-point",
                         options={"maxiter": 1000, "disp": False, "tol": 1.e-9})

        if Min_cf.status == 0:
            self.result = Min_cf.x.copy()
            self.result = self.result * self.scale
            self.result = self.result.round(0)
            self.result = self.result / self.scale
            self.optimized_function_value = round(Min_cf.fun, 0)
            self.solution_message = Min_cf.message
        else:
            Min_cf = linprog(cost_function, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                             method="revised simplex",
                             options={"maxiter": 1000, "disp": False, "tol": 1.e-9})
            if Min_cf.status == 0:
                self.result = Min_cf.x.copy()
                self.result = self.result * self.scale
                self.result = self.result.round(0)
                self.result = self.result / self.scale
                self.optimized_function_value = round(Min_cf.fun, 0)
                self.solution_message = Min_cf.message
        if Min_cf.status == 0:
            self.result_status = "success"
            string3 = "The Current Iteration Number: " + str(Min_cf.nit)
            string2 = "The Current Function Value: " + str(self.optimized_function_value)
            self.solution_message = Min_cf.message + "\n" + string2 + "\n" + string3
            capa = self.f_capacity_constraint[:no_of_f].reshape((1, no_of_f))
            demand = self.c_demand_constraint[:no_of_c].reshape((1, no_of_c))
            shipment = self.result[:no_of_decision_variable].reshape((no_of_f, no_of_c))
            self.df_shipment = pd.DataFrame(shipment, self.data_frame_index[0], self.data_frame_index[1])
            total_shipped_to_c = np.sum(shipment, axis=0)
            total_shipped_to_c = total_shipped_to_c.reshape((1,no_of_c))
            total_shipped_from_f = np.sum(shipment, axis=1)
            total_shipped_from_f = total_shipped_from_f.reshape((1,no_of_f))
            Slack_f = capa - total_shipped_from_f
            Slack_c = total_shipped_to_c - demand
            df_np = np.append(total_shipped_from_f, capa, axis=0)
            df_np = np.append(df_np, Slack_f, axis=0)
            self.df_c_slack = pd.DataFrame(df_np, ["Total Shipment", "Capacity", "Slack"], self.data_frame_index[0])
            df_np = np.append(total_shipped_to_c, demand, axis=0)
            df_np = np.append(df_np, Slack_c, axis=0)
            self.df_d_slack = pd.DataFrame(df_np, ["Total Shipment", "Demand", "Slack"], self.data_frame_index[1])
        else:
            self.result_status = "not_success"
            self.solution_message = Min_cf.message


problem = TransPortationModel()


def generate_table(df):
    h_style = {"border": "2px solid blue", "text-align": "left", "background": "#f1f1c1", "color": "black"}
    b_style = {"border": "2px solid blue", "text-align": "left", "color": "black", "background": "white"}
    df_np = np.array(df)
    row = df.shape[0]
    column = df.shape[1]
    temp = df.columns.values.tolist()
    table_head = [np.nan]
    for i in range(0,column):
        table_head.append(temp[i])
    temp = np.array(df.index.tolist()).reshape((row, 1))
    table_df = np.append(temp, df_np, axis=1)
    return html.Table(
                      children=[html.Thead(
                                        html.Tr([html.Th(table_head[col],style=h_style) for col in range(0,column+1)])
                                           ),
                                html.Tbody([html.Tr([html.Td(table_df[i,j],style=b_style) for j in range(0,column+1)]
                                                      ) for i in range(0,row)]
                                              )
                                ],style={"width":"100%"}
                      )


tool_tip = """Three separate excel files(namely 1-cost.xls, 2-capacity.xls and 3-demand.xls) 
              should be selected for shipment cost,factories capacity and customers demand. 
              The format of excel file should be same as the data table shown in interactive window.
            """
m_description = """This model consists of shipment from sources to destinations with single commodity.
                The objective of the solution is to minimize the costs of shipping goods from sources
                to destination, while not exceeding the supply available from each source and meeting
                the demand of each destination. 
                Input of the model are Number of factories(Sources),Number of customers(Destinations), 
                Shipment cost per unit of commodities from each factory to each customer, Production capacity  
                of commodities in units at each factory and Demand of commodities in units at each customer.   
                Two options are available for giving input the first one is using interactive sheets the other 
                is uploading data by excel sheets.
                """

rows = html.Div([
    html.Div([
        dbc.Row([dbc.Col(dbc.Alert(html.H4("Sigle-Stage-Single-Commodity Transportation Model"),color="primary"),width=8),
                 dbc.Col(dbc.Button("Model Description", id="Model-Description-B", n_clicks=0, color="info"),
                 width={"size": "auto", "offset": 2})]),
        dbc.Modal([
            dbc.ModalHeader("Model Description"),
            dbc.ModalBody(m_description,style={"color": "blue"}),
            dbc.ModalFooter(dbc.Button("Close", id="Modal-Description-close", n_clicks=0))
        ], id="Modal-Description")
    ]),

    html.Div([
    dbc.Row([dbc.Col(dbc.Alert("Define Transportation Model Size", color="primary"), width=5),
             dbc.Col(
                 dcc.Upload(id="upload",
                 children=[dbc.Button("Upload Excel Files",id="upload-button", n_clicks=0, color="success")
                     ,dbc.Tooltip(tool_tip,target="upload-button",placement="bottom")],multiple= True),
                 width={"size": "auto", "offset": 5})]),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupAddon("Number Of Factories", addon_type="prepend"),
    dbc.Input(id="dim-1", placeholder="Enter Here.", type="number", min=1, step=1)]), width=4)]),
    html.Br(),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupAddon("Number Of Customers", addon_type="prepend"),
    dbc.Input(id="dim-2", placeholder="Enter Here.", type="number", min=1, step=1)]), width=4)]),
    html.Br(),
    dbc.Row([dbc.Col(dbc.Button("Submit", id="dim-enter", n_clicks=0, color="success", className="mr-1"),
                                                                       width={"size": 2, "offset": 5})]),
    dbc.Modal([
    dbc.ModalHeader("FYI"),
    dbc.ModalBody("Empty Data Fields Are Found",id="modal-0-content",style={"color":"blue"}),
    dbc.ModalFooter(dbc.Button("Close",id = "close-modal-0",n_clicks=0) )
    ],id="modal-0"),
    html.Br(),
    ],style={"background":"grey","border":"2px blue solid"}),

    html.Div([
    dbc.Row([dbc.Col(dbc.Alert("Shipment Cost/Unit From Factories To Customers",color="primary"), width=5),]),
    dbc.Row([dbc.Col(
    dash_table.DataTable(id="table-1",
                             columns=[],
                             data=[],
                             editable=True,
                             export_format="xlsx",
                             style_header={"border":"2px solid blue","color":"black","background":"#f1f1c1"},
                             style_data={"border":"2px solid blue","color":"black", "background": "white"}),
                              width={"size": "auto", "offset": 1}) ]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("Customers Demand in Unit",color="primary"), width=5),]),
    dbc.Row([dbc.Col(
    dash_table.DataTable(id="table-2",
                             columns=[],
                             data=[],
                             editable=True,
                             export_format="xlsx",
                             style_header={"border": "2px solid blue", "color": "black", "background": "#f1f1c1"},
                             style_data={"border": "2px solid blue", "color": "black", "background": "white"}),
                             width={"size": "auto", "offset": 1})]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("Factories Capacity in Unit",color="primary"), width=5),]),
    dbc.Row([dbc.Col(
    dash_table.DataTable(id="table-3",
                             columns=[],
                             data=[],
                             editable=True,
                             export_format="xlsx",
                             style_header={"border": "2px solid blue", "color": "black", "background": "#f1f1c1"},
                             style_data={"border": "2px solid blue", "color": "black", "background": "white"}),
                             width={"size": "auto", "offset": 1})]),
     html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("Define Input Units", color="primary"), width=5), ]),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupAddon("Scale,1-Unit=", addon_type="prepend"),
    dbc.Input(id="scale", placeholder="Quantity Of Commodities?", type="number", min=1, step=1)]), width=4)]),
    html.Br(),
    dbc.Row([dbc.Col(dbc.Button("Solve", id="solve", n_clicks=0, color="success"),
                             width={"size": 2, "offset": 5})]),
    dbc.Modal([
    dbc.ModalHeader("Solution Status"),
    dbc.ModalBody("Empty Data Fields Are Found",id="modal-1-content",style={"color":"blue"}),
    dbc.ModalFooter(dbc.Button("Close",id = "close-modal-1",n_clicks=0) )
    ],id="modal-1"),
    html.Br(),
    ],id="hide-1",style={"background":"grey","border":"2px blue solid","overflowX": 'auto','display': 'none'}),


    html.Div([
    dbc.Row([
    dbc.Col(dbc.Alert("The Total Optimum Shipment Cost" , color="primary"), width=5),]),

    dbc.Row([dbc.Col(html.Div([],id="result-tc"), width={"size": "auto", "offset": 1}),]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("Shipment From Factories To Customers", color="primary"), width=5)]),
    dbc.Row([dbc.Col(html.Div([], id="result-shipment"), width={"size": "auto", "offset": 1}),]),
    html.Hr(),
    html.Hr(),
    dbc.Row([
    dbc.Col(dbc.Button("Shipment Vs Capacity", id="constraint-1", n_clicks=0, color="success",
                                                                ), width= "auto"),
    dbc.Col(dbc.Button("Shipment Vs Demand", id="constraint-2", n_clicks=0, color="success",
                                                              ), width= "auto"),
    ])],id="hide-2",style={"background":"grey","border":"5px blue solid","overflowX": 'auto','display': 'none'}),


    html.Div([
    html.Div([],id="result-title-1"),
    dbc.Row([dbc.Col(html.Div([], id="result-item"), width={"size": "auto", "offset": 1}),]),
    html.Hr(),
    dcc.Graph(id="graph-1"),
    ],id="hide-3",style={"background":"grey","border":"5px blue solid","overflowX": 'auto','display': 'none'}),

    html.Div([html.Br() for i in range(0,10)])

    ])
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([rows])


@app.callback(
    dash.dependencies.Output("Modal-Description", "is_open"),
    [dash.dependencies.Input("Model-Description-B", "n_clicks"),
     dash.dependencies.Input("Modal-Description-close", "n_clicks")],
    [dash.dependencies.State("Modal-Description", "is_open")],
)
def model_description(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


def hide_one(is_open,msg):
    op_list = []
    style = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'none'}
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append(style)
    op_list.append(not is_open)
    op_list.append(msg)
    op_list.append(None)
    op_list.append(None)
    return op_list


def show_one(is_open):
    problem.create_data_frames()
    op_list = []
    style = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'block'}
    op_list.append(problem.df_cost.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_cost.columns])
    op_list.append(problem.df_demand.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_demand.columns])
    op_list.append(problem.df_capacity.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_capacity.columns])
    op_list.append(style)
    op_list.append(is_open)
    op_list.append(" ")
    op_list.append(problem.no_of_f)
    op_list.append(problem.no_of_c)
    return op_list


@app.callback(
    [dash.dependencies.Output("table-1", "data"),
     dash.dependencies.Output("table-1", "columns"),
     dash.dependencies.Output("table-2", "data"),
     dash.dependencies.Output("table-2", "columns"),
     dash.dependencies.Output("table-3", "data"),
     dash.dependencies.Output("table-3", "columns"),
     dash.dependencies.Output("hide-1", "style"),
     dash.dependencies.Output("modal-0", "is_open"),
     dash.dependencies.Output("modal-0-content", "children"),
     dash.dependencies.Output("dim-1", "value"),
     dash.dependencies.Output("dim-2", "value"),],
    [dash.dependencies.Input("dim-enter",  "n_clicks"),
     dash.dependencies.Input("upload", 'contents'),
     dash.dependencies.Input("close-modal-0", "n_clicks"),
     dash.dependencies.State("dim-1", "value"),
     dash.dependencies.State("dim-2", "value"),
     dash.dependencies.State("upload", 'filename'),
     dash.dependencies.State("modal-0", "is_open")
    ])
def size_enter(n1,list_of_contents,n2,v1,v2,list_of_names,is_open):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if "dim-enter" in changed_id:
        if v1 == None or v2 == None:
            return hide_one(is_open,"Invalid Input")
        elif v1 < 1 or v2 < 1:
            return hide_one(is_open,"Invalid Input")
        else:
            problem.no_of_f = int(v1)
            problem.no_of_c = int(v2)
            problem.index_headers = ["Factories"]
            string1 = "Customer-"
            string2 = "Factory-"
            problem.facility_names.clear()
            for i in range(0, problem.no_of_f):
                problem.facility_names.append(string2 + str(i + 1))
            problem.customer_names.clear()
            for i in range(0, problem.no_of_c):
                problem.customer_names.append(string1 + str(i + 1))
            problem.make_me_none()
            return show_one(is_open)
    elif "upload" in changed_id:
        temp =[False]*3
        for i in list_of_names:
            if "xls" in i:
                temp.append(True)
            else:
                temp.append(False)
            if "cost" in i:
                temp[0] = True
            elif "demand" in i:
                temp[1] = True
            elif "capacity" in i:
                temp[2] = True
        mask = np.array(temp)
        mask = mask.all()
        if len(list_of_contents) == 3 and mask:
            dict_cont = dict(zip(list_of_names, list_of_contents))
            for i, j in dict_cont.items():
                if "cost" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_cost = pd.read_excel(io.BytesIO(decoded))
                    headers = df_cost.columns.values.tolist()
                    problem.index_headers = headers[0:1].copy()
                    h1 = df_cost[headers[0]]
                    h1_np = np.array(h1)
                    problem.no_of_f = h1_np.size
                    problem.facility_names = h1_np.tolist()
                    df_cost.drop(headers[0], axis=1, inplace=True)
                    problem.customer_names = df_cost.columns.values.tolist()
                    problem.no_of_c = len(problem.customer_names)
                    problem.cost_for_f_to_cust = np.array(df_cost).flatten()

                elif "demand" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_demand = pd.read_excel(io.BytesIO(decoded))
                    problem.c_demand_constraint = np.array(df_demand).flatten()
                elif "capacity" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_capacity = pd.read_excel(io.BytesIO(decoded))
                    problem.f_capacity_constraint = np.array(df_capacity).flatten()
            return show_one(is_open)
        else:
            msg = """Three excel files should be selected.
            Names of excel files should be like 1-cost.xls, 2-capacity.xls, 3-demand.xls"""
            return hide_one(is_open, msg)

    elif "close-modal-0" in changed_id:
        msg = " "
        return hide_one(is_open,msg)
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output("result-tc", "children"),
     dash.dependencies.Output("result-shipment", "children"),
     dash.dependencies.Output("hide-2", "style"),
     dash.dependencies.Output("modal-1", "is_open"),
     dash.dependencies.Output("modal-1-content", "children"),
     dash.dependencies.Output("scale", "value")],
    [dash.dependencies.Input("solve",  "n_clicks"),
    dash.dependencies.Input("close-modal-1", "n_clicks"),
    dash.dependencies.State("table-1", "data"),
    dash.dependencies.State("table-2", "data"),
    dash.dependencies.State("table-3", "data"),
    dash.dependencies.State("scale", "value"),
    dash.dependencies.State("modal-1", "is_open")
     ])
def solve_model(n1,n2,cost,demand,capa,u,is_open):
    if n1 == 0 and n2 == 0:
        raise PreventUpdate
    else:
        if u == None:
            u = 1
            problem.scale = 1
        elif u <= 0:
            u = 1
            problem.scale = 1
        else:
            problem.scale = u
        df1 = pd.DataFrame(cost)
        headers = df1.columns.values.tolist()
        df1.drop(headers[0],axis=1,inplace=True)
        problem.cost_for_f_to_cust = np.array(df1).flatten().astype(np.float)
        df2 = pd.DataFrame(demand)
        problem.c_demand_constraint = np.array(df2).flatten().astype(np.float)
        df3 = pd.DataFrame(capa)
        problem.f_capacity_constraint = np.array(df3).flatten().astype(np.float)

        mask1 = np.isnan(problem.cost_for_f_to_cust)
        mask1 = np.invert(mask1).all()
        mask2 = np.isnan(problem.c_demand_constraint)
        mask2 = np.invert(mask2).all()
        mask3 = np.isnan(problem.f_capacity_constraint)
        mask3 = np.invert(mask3).all()
        result1 = html.Div()
        table = html.Div()
        if mask1 and mask2 and mask3:
            if not is_open:
                problem.solve_problem()
            if problem.result_status == "success":
                string1 = str(problem.optimized_function_value)
                result1 = html.H4(string1)
                table = generate_table(problem.df_shipment)
                style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'block'}
            else:
                style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'none'}

            return result1, table, style, not is_open, problem.solution_message,u
        else:
            style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'none'}
            return result1, table, style, not is_open,"Empty Data Fields Are Found",u


@app.callback(
    [dash.dependencies.Output("result-title-1", "children"),
     dash.dependencies.Output("result-item", "children"),
     dash.dependencies.Output("hide-3", "style"),
     dash.dependencies.Output("graph-1", "figure")],
    [dash.dependencies.Input("constraint-1",  "n_clicks"),
     dash.dependencies.Input("constraint-2",  "n_clicks"),
     ])
def problem_results(n1,n2):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    colors = {"background1": "#111111", "background2": "#f1f1c1", "background3": "gray", "text1": "white","text2": "blue"}
    style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'block'}
    if "constraint-1" in changed_id:
        title= dbc.Row([
        dbc.Col(dbc.Alert("Shipment From Factories Vs Capacity", color="primary"), width=5), ])
        table = generate_table(problem.df_c_slack)
        df = problem.df_c_slack.transpose()
        dict1 = df.to_dict("split")
        index = dict1["index"]
        figure = {"data": [
            {"x": index, "y": df["Total Shipment"], "type": "bar", "name": "Shipment"},
            {"x": index, "y": df["Capacity"], "type": "bar", "name": "Capacity"}
        ],
            "layout": {
                "title": "Shipment Vs Capacity",
                "plot_bgcolor": colors["background1"],
                "paper_bgcolor": colors["background3"],
                "font_color": colors["text2"]
                       }}
        return title, table, style, figure
    elif "constraint-2" in changed_id:
        title= dbc.Row([
        dbc.Col(dbc.Alert("Shipment To Customers Vs Demand", color="primary"), width=5), ])
        table = generate_table(problem.df_d_slack)
        df = problem.df_d_slack.transpose()
        dict1 = df.to_dict("split")
        index = dict1["index"]
        figure = {"data": [
            {"x": index, "y": df["Total Shipment"], "type": "bar", "name": "Shipment"},
            {"x": index, "y": df["Demand"], "type": "bar", "name": "Demand"}
        ],
            "layout": {
                "title": "Shipment Vs Demand",
                "plot_bgcolor": colors["background1"],
                "paper_bgcolor": colors["background3"],
                "font_color": colors["text2"]
            }}
        return title, table, style, figure

    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)

