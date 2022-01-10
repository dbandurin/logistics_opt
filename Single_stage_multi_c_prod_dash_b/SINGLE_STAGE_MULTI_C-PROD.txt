
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
        self.no_of_p = 1
        self.scale = 1
        self.product_code = []
        self.product_code_size = 1
        self.unique_product_code = np.array([np.nan])
        self.facility_names = []
        self.customer_names = []
        self.index_headers = []

    def make_me_none(self):
        size = problem.product_code_size * self.no_of_c
        temp = [np.nan] * size
        self.cost_for_production = np.array(temp)
        self.cost_for_f_to_cust = np.array(temp)
        self.f_capacity_constraint = np.array(temp)
        self.c_demand_constraint = np.array(temp)

    def create_data_frames(self):
        no_of_f = self.no_of_f
        no_of_c = self.no_of_c
        self.data_frame_index = []
        temp1 = []
        temp2 = []
        for i in range(0, no_of_f):
            size = len(self.product_code[i])
            for j in range(0, size):
                temp1.append(self.facility_names[i])
                temp2.append(self.product_code[i][j])
        index = list(zip(temp1, temp2))
        index = pd.MultiIndex.from_tuples(index)
        self.data_frame_index.append(index)
        self.data_frame_index.append(self.customer_names)
        self.data_frame_index.append(self.unique_product_code)
        temp1 = []
        temp2 = []
        for i in range(0, self.no_of_c):
            size = self.unique_product_code.size
            for j in range(0, size):
                temp1.append(self.customer_names[i])
                temp2.append(self.unique_product_code[j])
        index = list(zip(temp1, temp2))
        index = pd.MultiIndex.from_tuples(index)
        self.data_frame_index.append(index)

        temp1 = []
        temp2 = []
        temp = ["Shipment","Demand","Slack"]
        for i in range(0, 3):
            size = self.unique_product_code.size
            for j in range(0, size):
                temp1.append(temp[i])
                temp2.append(self.unique_product_code[j])
        index = list(zip(temp1, temp2))
        index = pd.MultiIndex.from_tuples(index)
        self.data_frame_index.append(index)

        row_size = self.product_code_size
        p_size = self.unique_product_code.size
        demand_size = p_size * no_of_c
        shipment_size = self.product_code_size * no_of_c
        p_cost = self.cost_for_production[:row_size].reshape((row_size, 1))
        cost = self.cost_for_f_to_cust[:shipment_size].reshape((row_size, no_of_c))
        capa = self.f_capacity_constraint[:row_size].reshape((row_size, 1))
        demand = self.c_demand_constraint[:demand_size].reshape((no_of_c, p_size))
        demand = demand.transpose()
        f_index=[]
        for i in range(0,no_of_f):
            f_index.append(self.facility_names[i])
            size = len(self.product_code[i])
            for j in range(1,size):
                f_index.append(np.nan)
        p_index = []
        for i in range(0, no_of_f):
            size = len(self.product_code[i])
            for j in range(0, size):
                p_index.append(self.product_code[i][j])

        self.df_prod_cost = pd.DataFrame(p_cost, p_index, ["Production Cost"])
        df = pd.DataFrame(p_index, p_index, [self.index_headers[1]])
        self.df_prod_cost = pd.concat([df, self.df_prod_cost], axis=1)
        df = pd.DataFrame(f_index, p_index, [self.index_headers[0]])
        self.df_prod_cost = pd.concat([df, self.df_prod_cost], axis=1)

        self.df_cost = pd.DataFrame(cost,p_index,self.data_frame_index[1])
        df=pd.DataFrame(p_index,p_index,[self.index_headers[1]])
        self.df_cost = pd.concat([df, self.df_cost], axis=1)
        df = pd.DataFrame(f_index,p_index,[self.index_headers[0]])
        self.df_cost = pd.concat([df, self.df_cost], axis=1)

        self.df_demand = pd.DataFrame(demand, self.data_frame_index[2],self.data_frame_index[1])
        df = pd.DataFrame(self.data_frame_index[2], self.data_frame_index[2], [self.index_headers[1]])
        self.df_demand = pd.concat([df, self.df_demand], axis=1)

        self.df_capacity = pd.DataFrame(capa, p_index,["Capacity"])
        df = pd.DataFrame(p_index, p_index, [self.index_headers[1]])
        self.df_capacity = pd.concat([df, self.df_capacity], axis=1)
        df = pd.DataFrame(f_index, p_index, [self.index_headers[0]])
        self.df_capacity = pd.concat([df, self.df_capacity], axis=1)

    def solve_problem(self):
        no_of_f = self.no_of_f
        no_of_c = self.no_of_c
        p_size = self.product_code_size
        unique_size = self.unique_product_code.size
        demand_size = unique_size * no_of_c
        no_of_decision_variable = p_size * no_of_c

        A_ub = np.zeros((p_size, no_of_c))
        A_ub[0] = 1
        A_ub = A_ub.reshape(no_of_decision_variable)
        for i in range(1, p_size):
            z = np.zeros((p_size, no_of_c))
            z[i] = 1
            z = z.reshape(no_of_decision_variable)
            A_ub = np.vstack((A_ub, z))
        temp = self.unique_product_code.copy()
        unique_product_code_list = temp.tolist()

        for i in range(0, no_of_c):
            for j in unique_product_code_list:
                f_to_c = np.zeros((p_size, no_of_c), dtype=int)
                s = 0
                for k in range(0, no_of_f):
                    if j in self.product_code[k]:
                        pos = self.product_code[k].index(j)
                        f_to_c[s + pos, i] = -1
                    s = s + len(self.product_code[k])

                f_to_c_flatten = f_to_c.flatten()
                A_ub = np.vstack((A_ub, f_to_c_flatten))
        demand_constraint_neg = -1 * self.c_demand_constraint
        b_ub = np.append(self.f_capacity_constraint, demand_constraint_neg)

        cost_function = self.cost_for_f_to_cust.copy()
        cost_function = cost_function.reshape((p_size, no_of_c))
        temp = self.cost_for_production[:p_size].copy()
        temp = temp.reshape((p_size, 1))
        cost_function = cost_function + temp
        cost_function = cost_function.flatten()
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
            capa = self.f_capacity_constraint[:p_size].reshape((p_size,1))
            demand = self.c_demand_constraint[:demand_size].reshape((no_of_c, unique_size))
            demand = demand.transpose()
            demand_graph = self.c_demand_constraint[:demand_size].reshape((demand_size, 1))
            shipment = self.result[:no_of_decision_variable].reshape((p_size, no_of_c))
            self.df_shipment = pd.DataFrame(shipment, self.data_frame_index[0], self.data_frame_index[1])
            self.df_shipment.index.names = self.index_headers
            total_shipped_from_f = np.sum(shipment, axis=1)
            total_shipped_from_f = total_shipped_from_f.reshape((p_size,1))
            Slack_f = capa - total_shipped_from_f
            df_np = np.append(total_shipped_from_f, capa, axis=1)
            df_np = np.append(df_np, Slack_f, axis=1)
            self.df_c_slack = pd.DataFrame(df_np,  self.data_frame_index[0],["Total Shipment", "Capacity", "Slack"])
            self.df_c_slack.index.names = self.index_headers

            to_customer = self.df_shipment.sum(level=1, axis=0)
            to_customer_np = np.array(to_customer)
            slack = to_customer_np - demand
            demand_slack = np.append(to_customer, demand, axis=0)
            demand_slack = np.append(demand_slack, slack, axis=0)
            self.df_d_slack = pd.DataFrame(demand_slack,self.data_frame_index[4],self.data_frame_index[1])
            self.df_d_slack.index.names = ["", "Products"]

            to_customer = self.df_shipment.sum(level=1, axis=0)
            to_customer_np = np.array(to_customer).flatten(order="f").reshape((demand_size, 1))
            to_customer_np = np.append(to_customer_np, demand_graph, axis=1)
            self.df_demand_graph = pd.DataFrame(to_customer_np, self.data_frame_index[3], ["Shipment", "Demand"])
        else:
            self.result_status = "not_success"
            self.solution_message = Min_cf.message


problem = TransPortationModel()


def generate_m_table(df):
    h_style = {"border": "2px solid blue", "text-align": "left", "background": "#f1f1c1", "color": "black"}
    b_style = {"border": "2px solid blue", "text-align": "left", "color": "black", "background": "white"}
    df_np = np.array(df)
    no_of_h = df.shape[1]
    no_of_row = df.shape[0]
    temp1 = df.columns.values.tolist()
    temp2 = [np.nan,np.nan]
    index_name = df.index.names.copy()
    for i in range(0, no_of_h):
        temp2.append(temp1[i])
        index_name.append(np.nan)
    table_head = []
    table_head.append(temp2)
    table_head.append(index_name)
    index_df = df.index.tolist()
    main_index = []
    sub_index = []
    for i in range(0, len(index_df)):
        if index_df[i][0] not in main_index:
            main_index.append(index_df[i][0])
            temp = []
            temp.append(index_df[i][1])
            for j in range(i + 1, len(index_df)):
                if index_df[i][0] == index_df[j][0]:
                    temp.append(index_df[j][1])
            sub_index.append(temp)
    no_of_mi = len(sub_index)

    content = []
    col_size = []
    step = 0
    for i in range(0,no_of_mi):
        temp = []
        size = len(sub_index[i])
        temp.append(main_index[i])
        temp.append(str(size))
        temp.append(h_style)
        count = 1
        for j in range(0,size):
            temp.append(sub_index[i][j])
            temp.append(str(1))
            temp.append(h_style)
            count = count + 1
            for k in range(0, no_of_h):
                temp.append(df_np[step + j,k])
                temp.append(str(1))
                temp.append(b_style)
                count = count + 1
            content.append(temp)
            col_size.append(count)
            count = 0
            temp = []
        step = step + size

    return html.Table(
        children=[html.Thead([
        html.Tr([html.Th(table_head[i][j], style=h_style) for j in range(0, no_of_h + 2)])for i in range(0, 2)]),
        html.Tbody([html.Tr([html.Td(content[i][j], rowSpan=content[i][j+1], style=content[i][j+2])
                                       for j in range(0, (col_size[i]*3),3)]) for i in range(0, no_of_row)])
        ], style={"width": "100%", "border-collapse": "collapse"})


tool_tip = """Four separate excel files(namely 1-shipment_cost.xls, 2-production_cost.xls, 3-capacity.xls and 4-demand.xls) 
              should be selected for shipment cost, production cost,factories capacity and customers demand. 
              The format of excel file should be same as the data table shown in interactive window.
            """
m_description = """This model consists of shipment from sources to destinations with multi commodities.
                The objective of the solution is to minimize the costs of shipping goods from sources to destination
                with considering production cost, while not exceeding the supply available from each source and meeting
                the demand of each destination. 
                Input of the model are Number of factories(Sources),Number of customers(Destinations), 
                Commodity list of each factories,Shipment cost per unit of commodities from each factory to each 
                customer,Production cost of commodities at each factories , Production capacity  of commodities in units
                at each factory and Demand of commodities in units at each customer. 
                Two options are available for giving input the first one is using interactive sheets the other is
                uploading data by excel sheets."""

rows = html.Div([
    html.Div([
        dbc.Row([dbc.Col(dbc.Alert(html.H4("Sigle-Stage-Multi-Commodity Transportation Cum Production Model"),color="primary"),width=9),
                 dbc.Col(dbc.Button("Model Description", id="Model-Description-B", n_clicks=0, color="info"),
                 width={"size": "auto", "offset": 1})]),
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
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupAddon("Number Of Products", addon_type="prepend"),
    dbc.Input(id="dim-3", placeholder="Enter Here.", type="number", min=1, step=1)]), width=4)]),
    html.Br(),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupAddon("Factories Have Same Product List?", addon_type="prepend"),
    dbc.Select(id="dim-select",options=[{"label":"Yes","value":"yes"},{"label":"No","value":"no"}],value="yes")]), width=4)]),
    html.Br(),
    dbc.Row([dbc.Col(dbc.Button("Submit", id="dim-submit", n_clicks=0, color="success", className="mr-1"),
                                                                       width={"size": 2, "offset": 5})]),
    dbc.Modal([
    dbc.ModalHeader("FYI"),
    dbc.ModalBody("Empty Data Fields Are Found",id="modal-0-content",style={"color":"blue"}),
    dbc.ModalFooter(dbc.Button("Close",id = "close-modal-0",n_clicks=0) )
    ],id="modal-0"),
    html.Br(),
    ],style={"background":"grey","border":"2px blue solid"}),

    html.Div([
        dbc.Row([dbc.Col(dbc.Alert("Product List Of Factories ", color="primary"), width=5), ]),
        dbc.Row([dbc.Col(
        dash_table.DataTable(id="table-0",
                            columns=[],
                            data=[],
                            editable=True,
                            style_table={"width":"100%"},
                            style_cell={"textAlign": "left"},
                            style_cell_conditional=[{"if": {"column_id":"Products"},"width":"80%"}],
                            style_header={"border": "2px solid blue", "color": "black", "background": "#f1f1c1"},
                            style_data={"border": "2px solid blue", "color": "black", "background": "white"}),
            width={"size":6, "offset": 1})]),
        html.Br(),
        dbc.Row([dbc.Col(dbc.Button("Submit", id="prod-submit", n_clicks=0, color="success", className="mr-1"),
                                                                       width={"size": 2, "offset": 5})]),
        html.Br(),
           ],id="hide-0",style={"background":"grey","border":"2px blue solid","overflowX": 'auto','display': 'none'}),

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
    dbc.Row([dbc.Col(dbc.Alert("Production Cost/Unit", color="primary"), width=5)]),
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
    dbc.Row([dbc.Col(dbc.Alert("Customers Demand in Units",color="primary"), width=5),]),
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
    dbc.Row([dbc.Col(dbc.Alert("Factories Capacity in Units",color="primary"), width=5),]),
    dbc.Row([dbc.Col(
    dash_table.DataTable(id="table-4",
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
    dbc.Row([dbc.Col(dbc.Alert("The Total Optimum Shipment Cost" , color="primary"), width=5),]),
    dbc.Row([dbc.Col(html.Div([],id="result-tc"), width={"size": "auto", "offset": 1}),]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("Shipment From Factories To Customers", color="primary"), width=5), ]),
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
    dbc.Row([dbc.Col(dbc.Alert("Shipment From Factories Vs Capacity", color="primary"), width=5)]),
    dbc.Row([dbc.Col(html.Div([], id="Shipment_Vs_Capacity"), width={"size": "auto", "offset": 1})]),
    html.Hr(),
    dcc.Graph(id="graph-1"),
    dbc.Row([dbc.Col(dcc.Slider(id="slider-1",min=0,step=None,marks={},value=0),width={"size": 10, "offset": 1})]),
    html.Br()
    ],id="hide-3",style={"background":"grey","border":"5px blue solid","overflowX": 'auto','display': 'none'}),

    html.Div([
    dbc.Row([dbc.Col(dbc.Alert("Shipment To Customers Vs Demand" , color="primary"), width=5)]),
    dbc.Row([dbc.Col(html.Div([], id="Shipment_Vs_Demand"), width={"size": "auto", "offset": 1})]),
    html.Hr(),
    dcc.Graph(id="graph-2"),
    dbc.Row([dbc.Col(dcc.Slider(id="slider-2", min=0, step=None, marks={}, value=0), width={"size": 10, "offset": 1})]),
    html.Br()
    ], id="hide-4", style={"background": "grey", "border": "5px blue solid", "overflowX": 'auto', 'display': 'none'}),

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


def hide_zero_and_one(is_open,msg):
    op_list = []
    style = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'none'}
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append(style)
    op_list.append(style)
    op_list.append(not is_open)
    op_list.append(msg)
    op_list.append(None)
    op_list.append(None)
    op_list.append(None)
    return op_list


def hide_zero_show_one(is_open):
    op_list = []
    style1 = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'block'}
    style0 = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'none'}
    problem.create_data_frames()
    op_list.append([])
    op_list.append([])
    op_list.append(problem.df_cost.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_cost.columns])
    op_list.append(problem.df_prod_cost.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_prod_cost.columns])
    op_list.append(problem.df_demand.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_demand.columns])
    op_list.append(problem.df_capacity.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_capacity.columns])
    op_list.append(style0)
    op_list.append(style1)
    op_list.append(is_open)
    op_list.append(" ")
    op_list.append(problem.no_of_f)
    op_list.append(problem.no_of_c)
    op_list.append(problem.unique_product_code.size)
    return op_list


def show_zero_hide_one(is_open):
    op_list = []
    style0 = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'block'}
    style1 = {"background": "grey", "border": "2px blue solid", "overflowX": "auto", 'display': 'none'}
    p_list = np.array(([np.nan] * problem.no_of_f)).reshape((problem.no_of_f, 1))
    df_p_list = pd.DataFrame(p_list, problem.facility_names, ["Product List(Enter by comma separators)"])
    df = pd.DataFrame(problem.facility_names, problem.facility_names, [problem.index_headers[0]])
    df_p_list = pd.concat([df, df_p_list], axis=1)
    op_list.append(df_p_list.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in df_p_list])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append([])
    op_list.append(style0)
    op_list.append(style1)
    op_list.append(is_open)
    op_list.append(" ")
    op_list.append(problem.no_of_f)
    op_list.append(problem.no_of_c)
    op_list.append(None)
    return op_list


@app.callback(
    [dash.dependencies.Output("table-0", "data"),
     dash.dependencies.Output("table-0", "columns"),
     dash.dependencies.Output("table-1", "data"),
     dash.dependencies.Output("table-1", "columns"),
     dash.dependencies.Output("table-2", "data"),
     dash.dependencies.Output("table-2", "columns"),
     dash.dependencies.Output("table-3", "data"),
     dash.dependencies.Output("table-3", "columns"),
     dash.dependencies.Output("table-4", "data"),
     dash.dependencies.Output("table-4", "columns"),
     dash.dependencies.Output("hide-0", "style"),
     dash.dependencies.Output("hide-1", "style"),
     dash.dependencies.Output("modal-0", "is_open"),
     dash.dependencies.Output("modal-0-content", "children"),
     dash.dependencies.Output("dim-1", "value"),
     dash.dependencies.Output("dim-2", "value"),
     dash.dependencies.Output("dim-3", "value")],
    [dash.dependencies.Input("dim-submit",  "n_clicks"),
     dash.dependencies.Input("prod-submit",  "n_clicks"),
     dash.dependencies.Input("upload", 'contents'),
     dash.dependencies.Input("close-modal-0", "n_clicks"),
     dash.dependencies.State("dim-1", "value"),
     dash.dependencies.State("dim-2", "value"),
     dash.dependencies.State("dim-3", "value"),
     dash.dependencies.State("dim-select", "value"),
     dash.dependencies.State("table-0", "data"),
     dash.dependencies.State("upload", 'filename'),
     dash.dependencies.State("modal-0", "is_open")])
def size_enter(n1,n2,list_of_contents,n3,v1,v2,v3,select,p_list,list_of_names,is_open):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if "dim-submit" in changed_id and select == "yes":
        if v1 == None or v2 == None or v3 == None:
            return hide_zero_and_one(is_open,"Invalid Input")
        elif v1 < 1 or v2 < 1 or v3 < 1:
            return hide_zero_and_one(is_open,"Invalid Input")
        else:
            problem.no_of_f = int(v1)
            problem.no_of_c = int(v2)
            problem.no_of_p = int(v3)
            problem.index_headers = ["Factories", "Products"]
            string1 = "Customer-"
            string2 = "Factory-"
            problem.facility_names.clear()
            for i in range(0, problem.no_of_f):
                problem.facility_names.append(string2 + str(i + 1))
            problem.customer_names.clear()
            for i in range(0, problem.no_of_c):
                problem.customer_names.append(string1 + str(i + 1))
            problem.product_code.clear()
            for i in range(0, problem.no_of_f):
                temp = []
                for j in range(1, problem.no_of_p + 1):
                    s = "Prod-" + str(j)
                    temp.append(s)
                problem.product_code.append(temp)
            temp = [item for sublist in problem.product_code for item in sublist]
            problem.product_code_size = len(temp)
            problem.unique_product_code = np.array(list(dict.fromkeys(temp)))
            problem.make_me_none()
            return hide_zero_show_one(is_open)
    elif "dim-submit" in changed_id and select == "no":
        if v1 == None or v2 == None:
            return hide_zero_and_one(is_open,"Invalid Input")
        elif v1 < 1 or v2 < 1:
            return hide_zero_and_one(is_open,"Invalid Input")
        else:
            problem.no_of_f = int(v1)
            problem.no_of_c = int(v2)
            problem.index_headers = ["Factories", "Products"]
            string1 = "Customer-"
            string2 = "Factory-"
            problem.facility_names.clear()
            for i in range(0, problem.no_of_f):
                problem.facility_names.append(string2 + str(i + 1))
            problem.customer_names.clear()
            for i in range(0, problem.no_of_c):
                problem.customer_names.append(string1 + str(i + 1))
            return show_zero_hide_one(is_open)
    elif "upload" in changed_id:
        temp =[False]*4
        for i in list_of_names:
            if "xls" in i:
                temp.append(True)
            else:
                temp.append(False)
            if "shipment_cost" in i:
                temp[0] = True
            elif "production_cost" in i:
                temp[1] = True
            elif "demand" in i:
                temp[2] = True
            elif "capacity" in i:
                temp[3] = True
        mask = np.array(temp)
        mask = mask.all()
        if len(list_of_contents) == 4 and mask:
            dict_cont = dict(zip(list_of_names, list_of_contents))
            for i, j in dict_cont.items():
                if "shipment_cost" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_shipmen_cost = pd.read_excel(io.BytesIO(decoded))
                    headers = df_shipmen_cost.columns.values.tolist()
                    problem.index_headers = headers[0:2].copy()
                    h1 = df_shipmen_cost[headers[0]].dropna()
                    h1_np = np.array(h1).tolist()
                    problem.facility_names = list(dict.fromkeys(h1_np))
                    problem.no_of_f = len(problem.facility_names)
                    h2 = df_shipmen_cost[headers[1]]
                    h2_np = np.array(h2)
                    problem.product_code_size = h2_np.size
                    h2_np = h2_np.tolist()
                    problem.unique_product_code = np.array(list(dict.fromkeys(h2_np)))
                    problem.no_of_p = problem.unique_product_code.size

                    h1 = df_shipmen_cost[headers[0]].fillna(value="same")
                    h1_np = np.array(h1)
                    for i in range(0,h1_np.size):
                        if h1_np[i] == "same":
                            h1_np[i] = h1_np[i-1]
                    h2 = df_shipmen_cost[headers[1]]
                    h2_np = np.array(h2)
                    fact_list = []
                    p_list = []
                    for i in range(0, h2_np.size):
                        if h1_np[i] not in fact_list:
                            fact_list.append(h1_np[i])
                            temp = []
                            temp.append(h2_np[i])
                            for j in range(i + 1, h2_np.size):
                                if h1_np[i] == h1_np[j]:
                                    temp.append(h2_np[j])
                            p_list.append(temp)
                    problem.product_code = p_list.copy()
                    df_shipmen_cost.drop([headers[0], headers[1]], axis=1, inplace=True)
                    problem.customer_names = df_shipmen_cost.columns.values.tolist()
                    problem.no_of_c = len(problem.customer_names)
                    problem.cost_for_f_to_cust = np.array(df_shipmen_cost).flatten()
                elif "production_cost" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_production_cost = pd.read_excel(io.BytesIO(decoded))
                    headers = df_production_cost.columns.values.tolist()
                    df_production_cost.drop([headers[0], headers[1]], axis=1, inplace=True)
                    problem.cost_for_production = np.array(df_production_cost).flatten()
                elif "demand" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_demand = pd.read_excel(io.BytesIO(decoded))
                    headers = df_demand.columns.values.tolist()
                    df_demand.drop(headers[0], axis=1, inplace=True)
                    df_demand = df_demand.transpose()
                    problem.c_demand_constraint = np.array(df_demand).flatten()
                elif "capacity" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_capacity = pd.read_excel(io.BytesIO(decoded))
                    headers = df_capacity.columns.values.tolist()
                    df_capacity.drop([headers[0], headers[1]], axis=1, inplace=True)
                    problem.f_capacity_constraint = np.array(df_capacity).flatten()
            return hide_zero_show_one(is_open)
        else:
            msg = """Four excel files should be selected.
            Names of excel files should be like 1-shipment_cost.xls, 2-production_cost.xls, 3-capacity.xls, 
            4-demand.xls"""
            return hide_zero_and_one(is_open,msg)
    elif "prod-submit" in changed_id:
        p_list_df = pd.DataFrame(p_list)
        mask = np.array(p_list_df["Product List(Enter by comma separators)"]) != None
        mask = mask.all()
        if mask:
            problem.product_code.clear()
            for i in range(0, problem.no_of_f):
                s = p_list_df.iloc[i]["Product List(Enter by comma separators)"]
                problem.product_code.append(s.split(","))
            temp = [item for sublist in problem.product_code for item in sublist]
            problem.product_code_size = len(temp)
            problem.unique_product_code = np.array(list(dict.fromkeys(temp)))
            problem.make_me_none()
            return hide_zero_show_one(is_open)
        else:
            return hide_zero_and_one(is_open,"Invalid Input")
    elif "close-modal-0" in changed_id:
        msg = " "
        return hide_zero_and_one(is_open,msg)
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
    dash.dependencies.State("table-4", "data"),
    dash.dependencies.State("scale", "value"),
    dash.dependencies.State("modal-1", "is_open")
     ])
def solve_model(n1,n2,shipment_cost,prod_cost,demand,capa,u,is_open):
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
        df1 = pd.DataFrame(shipment_cost)
        headers = df1.columns.values.tolist()
        df1.drop([headers[0],headers[1]],axis=1,inplace=True)
        problem.cost_for_f_to_cust = np.array(df1).flatten().astype(np.float)
        df2 = pd.DataFrame(prod_cost)
        headers = df2.columns.values.tolist()
        df2.drop([headers[0], headers[1]], axis=1, inplace=True)
        problem.cost_for_production = np.array(df2).flatten().astype(np.float)
        df3 = pd.DataFrame(demand)
        headers = df3.columns.values.tolist()
        df3.drop(headers[0], axis=1, inplace=True)
        df3 = df3.transpose()
        problem.c_demand_constraint = np.array(df3).flatten().astype(np.float)
        df4 = pd.DataFrame(capa)
        headers = df4.columns.values.tolist()
        df4.drop([headers[0], headers[1]], axis=1, inplace=True)
        problem.f_capacity_constraint = np.array(df4).flatten().astype(np.float)

        mask1 = np.isnan(problem.cost_for_f_to_cust)
        mask1 = np.invert(mask1).all()
        mask2 = np.isnan(problem.cost_for_production)
        mask2 = np.invert(mask2).all()
        mask3 = np.isnan(problem.c_demand_constraint)
        mask3 = np.invert(mask3).all()
        mask4 = np.isnan(problem.f_capacity_constraint)
        mask4 = np.invert(mask4).all()
        result1 = html.Div()
        table = html.Div()
        if mask1 and mask2 and mask3  and mask4:
            if not is_open:
                problem.solve_problem()
            if problem.result_status == "success":
                string1 = str(problem.optimized_function_value)
                result1 = html.H4(string1)
                table = generate_m_table(problem.df_shipment)
                style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'block'}
            else:
                style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'none'}

            return result1, table, style, not is_open, problem.solution_message,u
        else:
            style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'none'}
            return result1, table, style, not is_open,"Empty Data Fields Are Found",u


@app.callback(
    [dash.dependencies.Output("hide-3", "style"),
     dash.dependencies.Output("hide-4", "style"),
     dash.dependencies.Output("Shipment_Vs_Capacity", "children"),
     dash.dependencies.Output("Shipment_Vs_Demand", "children"),
     dash.dependencies.Output("graph-1", "figure"),
     dash.dependencies.Output("graph-2", "figure"),
     dash.dependencies.Output("slider-1",  "max"),
     dash.dependencies.Output("slider-1",  "marks"),
     dash.dependencies.Output("slider-2",  "max"),
     dash.dependencies.Output("slider-2",  "marks")],
    [dash.dependencies.Input("constraint-1",  "n_clicks"),
     dash.dependencies.Input("constraint-2",  "n_clicks"),
     dash.dependencies.Input("slider-1",  "value"),
     dash.dependencies.Input("slider-2",  "value")])
def check_constraints(n1,n2,v1,v2):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    colors = {"background1": "#111111", "background2": "#f1f1c1", "background3": "gray", "text1": "white","text2": "blue"}
    style_active = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'block'}
    style_hide = {"background": "grey", "border": "2px blue solid", "overflowX": 'auto', 'display': 'none'}

    if "constraint-1" in changed_id or "slider-1" in changed_id:
        g_title = "Shipment Vs Capacity For " + problem.facility_names[v1]
        table = generate_m_table(problem.df_c_slack)

        df = problem.df_c_slack.loc[problem.facility_names[v1]]
        dict1 = df.to_dict("split")
        index = dict1["index"]
        figure ={"data": [
            {"x": index, "y": df["Total Shipment"], "type": "bar", "name": "Shipment"},
            {"x": index, "y": df["Capacity"], "type": "bar", "name": "Capacity"}
        ],
            "layout": {
                "title": g_title,
                "plot_bgcolor": colors["background1"],
                "paper_bgcolor": colors["background3"],
                "font_color": colors["text2"]
                       }}
        marks = {i:{ "label":problem.facility_names[i],"style":{"color":"blue"}} for i in range(0,problem.no_of_f)}
        return style_active, style_hide, table, html.Div(), figure, {},problem.no_of_f, marks,problem.no_of_c,{}
    elif "constraint-2" in changed_id or "slider-2" in changed_id:
        g_title = "Shipment Vs Demand For " + problem.customer_names[v2]
        table = generate_m_table(problem.df_d_slack)
        df = problem.df_demand_graph.loc[problem.customer_names[v2]]
        dict1 = df.to_dict("split")
        index = dict1["index"]
        figure = {"data": [
            {"x": index, "y": df["Shipment"], "type": "bar", "name": "Shipment"},
            {"x": index, "y": df["Demand"], "type": "bar", "name": "Demand"}
        ],
            "layout": {
                "title": g_title,
                "plot_bgcolor": colors["background1"],
                "paper_bgcolor": colors["background3"],
                "font_color": colors["text2"]
            }}
        marks = {i: {"label": problem.customer_names[i], "style": {"color": "blue"}} for i in range(0, problem.no_of_c)}
        return style_hide, style_active, html.Div(), table, {}, figure,problem.no_of_f, {},problem.no_of_c,marks
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)

