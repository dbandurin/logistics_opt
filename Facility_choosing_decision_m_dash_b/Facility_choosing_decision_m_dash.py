
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import dash_table
from dash.exceptions import PreventUpdate
from scipy.optimize import minimize, linprog
from itertools import product
import base64
import io


class TransPortationModel:
    def __init__(self):
        self.no_of_f = 1
        self.no_of_w = 1
        self.no_of_p = 1
        self.scale = 1
        self.product_code = []
        self.product_code_size = 1
        self.unique_product_code = np.array([np.nan])
        self.facility_names = []
        self.warehouse_names = []
        self.index_headers = []
        self.open_close_decision = []
        self.combination_result_set = []
        self.combination_solve_set = []
        self.choosing_mode = 1

    def make_me_none(self):
        size = self.product_code_size * self.no_of_w
        temp = [np.nan] * size
        self.fixed_operating_cost = np.array(temp)
        self.cost_for_production = np.array(temp)
        self.cost_for_f_to_wh = np.array(temp)
        self.f_capacity_constraint = np.array(temp)
        self.w_demand_constraint = np.array(temp)

    def create_data_frames(self):
        no_of_f = self.no_of_f
        no_of_w = self.no_of_w
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
        self.data_frame_index.append(self.warehouse_names)
        self.data_frame_index.append(self.unique_product_code)
        temp1 = []
        temp2 = []
        for i in range(0, self.no_of_w):
            size = self.unique_product_code.size
            for j in range(0, size):
                temp1.append(self.warehouse_names[i])
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
        demand_size = p_size * no_of_w
        shipment_size = self.product_code_size * no_of_w
        f_cost = self.fixed_operating_cost[:no_of_f].reshape((no_of_f, 1))
        p_cost = self.cost_for_production[:row_size].reshape((row_size, 1))
        cost = self.cost_for_f_to_wh[:shipment_size].reshape((row_size, no_of_w))
        capa = self.f_capacity_constraint[:row_size].reshape((row_size, 1))
        demand = self.w_demand_constraint[:demand_size].reshape((no_of_w, p_size))
        demand = demand.transpose()
        f_m_index=[]
        f_index = []
        for i in range(0,no_of_f):
            f_index.append(self.facility_names[i])
            f_m_index.append(self.facility_names[i])
            size = len(self.product_code[i])
            for j in range(1,size):
                f_m_index.append(np.nan)
        p_index = []
        for i in range(0, no_of_f):
            size = len(self.product_code[i])
            for j in range(0, size):
                p_index.append(self.product_code[i][j])

        self.df_fixed_cost = pd.DataFrame(f_cost, f_index, ["Fixed Operation Cost"])
        df = pd.DataFrame(f_index, f_index, [self.index_headers[1]])
        self.df_fixed_cost = pd.concat([df, self.df_fixed_cost], axis=1)

        self.df_prod_cost = pd.DataFrame(p_cost, p_index, ["Production Cost"])
        df = pd.DataFrame(p_index, p_index, [self.index_headers[1]])
        self.df_prod_cost = pd.concat([df, self.df_prod_cost], axis=1)
        df = pd.DataFrame(f_m_index, p_index, [self.index_headers[0]])
        self.df_prod_cost = pd.concat([df, self.df_prod_cost], axis=1)

        self.df_cost = pd.DataFrame(cost,p_index,self.data_frame_index[1])
        df=pd.DataFrame(p_index,p_index,[self.index_headers[1]])
        self.df_cost = pd.concat([df, self.df_cost], axis=1)
        df = pd.DataFrame(f_m_index,p_index,[self.index_headers[0]])
        self.df_cost = pd.concat([df, self.df_cost], axis=1)

        self.df_demand = pd.DataFrame(demand, self.data_frame_index[2],self.data_frame_index[1])
        df = pd.DataFrame(self.data_frame_index[2], self.data_frame_index[2], [self.index_headers[1]])
        self.df_demand = pd.concat([df, self.df_demand], axis=1)

        self.df_capacity = pd.DataFrame(capa, p_index,["Capacity"])
        df = pd.DataFrame(p_index, p_index, [self.index_headers[1]])
        self.df_capacity = pd.concat([df, self.df_capacity], axis=1)
        df = pd.DataFrame(f_m_index, p_index, [self.index_headers[0]])
        self.df_capacity = pd.concat([df, self.df_capacity], axis=1)

    def solve_problem(self):
        open_close_d = []
        self.open_close_decision.clear()
        temp = self.unique_product_code.copy()
        unique_product_code_list = temp.tolist()
        no_of_w = self.no_of_w
        no_of_f = self.no_of_f
        prod_unique_size = self.unique_product_code.size
        unique_capacit_array = np.zeros((prod_unique_size * no_of_f), dtype=int)
        count = 0
        for j in unique_product_code_list:
            s1 = 0
            s2 = 0
            for k in range(0, no_of_f):
                if j in self.product_code[k]:
                    pos = self.product_code[k].index(j)
                    capa = self.f_capacity_constraint[pos + s1]
                    unique_capacit_array[count + s2] = capa
                s1 = s1 + len(self.product_code[k])
                s2 = s2 + prod_unique_size
            count = count + 1
        unique_capacit_array = unique_capacit_array.reshape((no_of_f, prod_unique_size))
        demand_array = self.w_demand_constraint.reshape((no_of_w,prod_unique_size))
        demand_array = demand_array.transpose()
        demand_array = np.cumsum(demand_array, axis=1)
        demand = demand_array[0:, no_of_w - 1]
        comb = product([1, 0], repeat=no_of_f)
        for i in list(comb):
            i = list(i)
            decesion_array = i * prod_unique_size
            decesion_array = np.array(decesion_array)
            decesion_array = decesion_array.reshape((no_of_f, prod_unique_size), order="f")
            production_array = unique_capacit_array * decesion_array
            production_array = np.cumsum(production_array, axis=0)
            production = production_array[no_of_f - 1, 0:]
            gate = production - demand
            bool_array = gate >= 0
            bool_array_gate = np.all(bool_array)
            if bool_array_gate:
                open_close_d.append(i)

        comb_array_size = len(open_close_d)
        if comb_array_size == 0:
            self.result_status = "not_success"
            self.solution_message = "No Feasible Solution For Meeting Current Warehouse Demand"
            return

        p_size = self.product_code_size
        trans_cost_array = self.cost_for_f_to_wh.reshape((p_size, no_of_w))
        p_cost = self.cost_for_production.reshape((p_size, 1))
        total_cost_array = p_cost + trans_cost_array
        self.combination_solve_set.clear()
        self.combination_result_set.clear()
        facility_list = []
        for i in range(0, no_of_f):
            facility_list.append(self.facility_names[i])
        facility_list = np.array(facility_list)
        si_set = []
        function_value_list = []
        operation_cost_list = []
        total_cost_list = []
        considered_facility_list = []
        selected_facility_list = []
        decision_variable_list = []
        count_comb = 0
        for comb in open_close_d:
            temp = []
            current_product_code = []
            for i in range(0, no_of_f):
                temp1 = []
                temp1.append(comb[i])
                temp1 = temp1 * len(self.product_code[i])
                temp.append(temp1)
                if comb[i] == 1:
                    current_product_code.append(self.product_code[i])
            current_no_of_f = len(current_product_code)
            temp2 = [item for sublist in current_product_code for item in sublist]
            current_p_size = len(temp2)
            mask = [item for sublist in temp for item in sublist]
            mask = np.array(mask)
            bool_mask = mask == 1
            temp = np.array(comb)
            comb_mask = temp == 1
            no_of_decision_variable = no_of_w * current_p_size
            current_total_cost_array = np.compress(bool_mask, total_cost_array, axis=0)
            A_ub = np.zeros_like(current_total_cost_array, dtype=int)
            A_ub[0] = 1
            A_ub = A_ub.reshape(no_of_decision_variable)
            for i in range(1, current_p_size):
                temp = np.zeros_like(current_total_cost_array, dtype=int)
                temp[i] = 1
                temp = temp.reshape(no_of_decision_variable)
                A_ub = np.vstack((A_ub, temp))

            temp = self.unique_product_code.copy()
            unique_product_code_list = temp.tolist()

            for i in range(0, no_of_w):
                for j in unique_product_code_list:
                    f_to_w = np.zeros_like(current_total_cost_array, dtype=int)
                    s = 0
                    for k in range(0, current_no_of_f):
                        if j in current_product_code[k]:
                            pos = current_product_code[k].index(j)
                            f_to_w[s + pos, i] = -1
                        s = s + len(current_product_code[k])
                    f_to_w_flatten = f_to_w.flatten()
                    A_ub = np.vstack((A_ub, f_to_w_flatten))

            current_f_capacity_constraint = self.f_capacity_constraint.reshape((p_size, 1))
            current_f_capacity_constraint = np.compress(bool_mask, current_f_capacity_constraint, axis=0)
            current_f_capacity_constraint = current_f_capacity_constraint.flatten()

            demand_constraint_neg = -1 * self.w_demand_constraint
            demand_constraint_neg = demand_constraint_neg.flatten()

            b_ub = np.append(current_f_capacity_constraint, demand_constraint_neg)

            cost_function = current_total_cost_array.flatten()

            temp = facility_list.reshape((no_of_f, 1))
            temp = np.compress(comb_mask, temp, axis=0)
            temp = temp.flatten()
            considered_facility_list.append(temp)

            Max_prod = [None] * no_of_decision_variable
            Min_prod = np.zeros(no_of_decision_variable)
            bounds = tuple(zip(Min_prod, Max_prod))

            Min_cf = linprog(cost_function, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                             method="interior-point",
                             options={"maxiter": 1000, "disp": False, "tol": 1.e-9})
            if Min_cf.status == 0:

                function_value_list.append(Min_cf.fun.round(0))
                decision_variable_list.append(Min_cf.x)
                self.solution_message = Min_cf.message
                self.result_status = "success"
            else:
                Min_cf = linprog(cost_function, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                                 method="revised simplex",
                                 options={"maxiter": 1000, "disp": False, "tol": 1.e-9})
                if Min_cf.status == 0:
                    function_value_list.append(Min_cf.fun.round(0))
                    decision_variable_list.append(Min_cf.x)
                    self.solution_message = Min_cf.message
                    self.result_status = "success"
                else:
                    self.solution_message = Min_cf.message
            if Min_cf.status == 0:
                self.open_close_decision.append(comb)
                current_product_code = []
                for i in range(0, no_of_f):
                    if comb[i] == 1:
                        current_product_code.append(self.product_code[i])
                n_row = int(Min_cf.x.size / no_of_w)
                result = Min_cf.x.copy()
                result = result * self.scale
                result = result.round(0)
                result = result / self.scale
                result = result.reshape((n_row, no_of_w))
                result = np.cumsum(result, axis=1)
                start = 0
                f_wise_sum = []
                for i in range(0, len(current_product_code)):
                    sum = 0
                    c_size = len(current_product_code[i])
                    for j in range(start, start + c_size):
                        sum = sum + result[j, no_of_w - 1]
                    f_wise_sum.append(sum)
                    start = start + c_size
                f_wise_sum = np.array(f_wise_sum)
                new_mask = f_wise_sum > 0
                temp = self.fixed_operating_cost.reshape((no_of_f, 1))
                temp = np.compress(comb_mask, temp, axis=0)
                temp = np.compress(new_mask, temp, axis=0)
                temp = temp.flatten().round(0)
                fixed_cost = temp.sum()
                operation_cost_list.append(fixed_cost)
                total_cost_list.append(Min_cf.fun.round(0) + fixed_cost)
                temp = facility_list.reshape((no_of_f, 1))
                temp = np.compress(comb_mask, temp, axis=0)
                temp = np.compress(new_mask, temp, axis=0)
                temp = temp.flatten()
                selected_facility_list.append(temp)
                si_set.append(count_comb)
                count_comb = count_comb + 1

        function_value_np = np.array(function_value_list)
        operation_cost_np = np.array(operation_cost_list)
        total_cost_np = np.array(total_cost_list)
        si_np = np.array(si_set)
        sorte_index = np.argsort(total_cost_np)
        temp = self.open_close_decision.copy()
        temp = np.array(temp)
        self.open_close_decision = temp[sorte_index].tolist()

        self.combination_result_set.append(function_value_np[sorte_index].tolist())
        self.combination_result_set.append(operation_cost_np[sorte_index].tolist())
        self.combination_result_set.append(total_cost_np[sorte_index].tolist())
        self.combination_result_set.append(si_np[sorte_index].tolist())
        self.combination_solve_set.append(decision_variable_list)
        self.combination_solve_set.append(considered_facility_list)
        self.combination_solve_set.append(selected_facility_list)

        c_size = len(self.open_close_decision)
        comb_list = []
        for i in range(0, c_size):
            comb_list.append("Comb-" + str(i + 1))
        o_c_decision = np.array(self.open_close_decision).flatten().reshape((c_size,no_of_f))
        self.df_combinations = pd.DataFrame(o_c_decision, comb_list, self.facility_names)

        tran_prod_cost = np.array(self.combination_result_set[0]).reshape((c_size,1))
        oper_cost = np.array(self.combination_result_set[1]).reshape((c_size, 1))
        total_cost = np.array(self.combination_result_set[2]).reshape((c_size, 1))
        result_list = np.append(tran_prod_cost,oper_cost,axis=1)
        result_list = np.append(result_list, total_cost, axis=1).astype(int)
        temp = ["Trans/Prod","Fix:Oper:","Total Cost"]
        df = pd.DataFrame(result_list, comb_list,temp)
        self.df_combinations = pd.concat([self.df_combinations, df], axis=1)
        self.result_pointer = 0

    def make_individual_results(self,pointer):
        self.result_pointer = pointer
        row_size = self.product_code_size
        u_size = self.unique_product_code.size
        no_of_f = self.no_of_f
        no_of_w = self.no_of_w
        w_row_size = u_size * no_of_w
        capa = self.f_capacity_constraint.reshape((row_size, 1))
        demand = self.w_demand_constraint.reshape((no_of_w, u_size))
        demand = demand.transpose()
        demand_graph = self.w_demand_constraint.reshape((w_row_size, 1))

        n = self.combination_result_set[3][self.result_pointer]
        self.result = self.combination_solve_set[0][n].round(0).copy()
        comb = self.open_close_decision[self.result_pointer].copy()
        f_list = self.combination_solve_set[1][n].copy()
        current_product_code = []
        temp = []
        for i in range(0, no_of_f):
            temp1 = []
            temp1.append(comb[i])
            temp1 = temp1 * len(self.product_code[i])
            temp.append(temp1)
            if comb[i] == 1:
                current_product_code.append(self.product_code[i])
        current_no_of_f = len(current_product_code)
        temp2 = [item for sublist in current_product_code for item in sublist]
        current_p_size = len(temp2)
        mask = [item for sublist in temp for item in sublist]
        mask = np.array(mask)
        bool_mask = mask == 1
        temp1 = []
        temp2 = []
        for i in range(0, current_no_of_f):
            size = len(current_product_code[i])
            for j in range(0, size):
                temp1.append(f_list[i])
                temp2.append(current_product_code[i][j])
        index = list(zip(temp1, temp2))
        current_index = pd.MultiIndex.from_tuples(index)
        size = current_p_size * no_of_w
        result = self.result.copy()
        result = result * self.scale
        result = result.round(0)
        result = result / self.scale
        shipment = result[:size].reshape((current_p_size, no_of_w))
        self.df_shipment = pd.DataFrame(shipment, current_index, self.data_frame_index[1])
        self.df_shipment.index.names = ["Facilities", "Products"]

        total_shipped_from_f = np.sum(shipment, axis=1)
        total_shipped_from_f = total_shipped_from_f.reshape((current_p_size, 1))
        capacity = capa.reshape((row_size, 1))
        capacity = np.compress(bool_mask, capacity, axis=0)
        Slack_f = capacity - total_shipped_from_f
        df_np6 = np.append(total_shipped_from_f, capacity, axis=1)
        df_np6 = np.append(df_np6, Slack_f, axis=1)
        self.df_capacity_slack = pd.DataFrame(df_np6, current_index, ["Total Shipment", "Capacity", "Slack"])
        self.df_capacity_slack.index.names = ["Facilities", "Products"]

        to_customer = self.df_shipment.sum(level=1, axis=0)
        to_customer_np = np.array(to_customer)
        slack = to_customer_np - demand
        demand_slack = np.append(to_customer, demand, axis=0)
        demand_slack = np.append(demand_slack, slack, axis=0)
        self.df_demand_slack = pd.DataFrame(demand_slack, self.data_frame_index[4], self.data_frame_index[1])
        self.df_demand_slack.index.names = ["", "Products"]

        df_temp = self.df_shipment.sum(level=1, axis=0)
        df_np0 = np.array(df_temp).flatten(order="f").reshape((w_row_size, 1))
        df_np0 = np.append(df_np0, demand_graph, axis=1)
        self.df_demand_slack_graph = pd.DataFrame(df_np0, self.data_frame_index[3], ["Shipment", "Demand"])
        self.df_demand_slack_graph["Slack"] = self.df_demand_slack_graph["Shipment"] - self.df_demand_slack_graph["Demand"]
        self.df_demand_slack_graph.index.names = ["Warehouses", "Products"]


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


tool_tip = """Five separate excel files(namely 1-shipment_cost, 2-fixed_operation_cost, 3-production_cost, 
              4-capacity and 5-demand.) should be selected for shipment cost,fixed operation cost, production cost,
              factories capacity and customers demand. The format of excel file should be same as the data table shown 
              in interactive window."""
m_description = """This model is used when we have to shutdown or choose some facilities in cases where the production
                capacity is much higher than the demand. Here we have to consider the total cost (Fixed operation cost +
                Production cost + Transportation cost) and do optimization to decide which facilities have to be
                chosen or shutdown ,while not exceeding the supply available from each chosen facilities and meeting
                the demand of each destination. 
                Input of the model are Number of facilities(Sources),Number of warehouses(Destinations), Commodity list
                of each facilities,Shipment cost per unit of commodities from each facility to each warehouses,Fixed 
                operation cost of each facilities,Production cost per unit  of commodities at each facilities,
                Production capacity of commodities in units at each facility and Demand of commodities in units at each
                warehouse. Two options are available for giving input the first one is using interactive sheets the 
                other is uploading data by excel sheets."""
t_description = """ Solution for all possible combination of chosen and unchosen combinations of facilities, based on
                warehouses demand, is listed in the table. One represent the chosen state and zero represent the unchosen
                state. You can go through the  Solution for each combination by selecting the checkbutton in every row 
                 """

rows = html.Div([
    html.Div([
        dbc.Row([dbc.Col(dbc.Alert(html.H4("Facility Choosing Decision Transportation Model"),color="primary"),width=9),
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
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupAddon("Number Of Warehouses", addon_type="prepend"),
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
    dbc.ModalFooter(dbc.Button("Close",id = "close-modal-0",n_clicks=0))
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
    dbc.Row([dbc.Col(dbc.Alert("Fixed Operation Cost", color="primary"), width=5)]),
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
    dbc.Row([dbc.Col(dbc.Alert("Production Cost/Unit", color="primary"), width=5)]),
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
    dbc.Row([dbc.Col(dbc.Alert("Warehouses Demand in Units",color="primary"), width=5),]),
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
    dbc.Row([dbc.Col(dbc.Alert("Factories Capacity in Units",color="primary"), width=5),]),
    dbc.Row([dbc.Col(
    dash_table.DataTable(id="table-5",
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
    dbc.Row([dbc.Col(dbc.Alert("Optimum Solution For All Possible Combinations", color="primary"), width=5),
             dbc.Col(dbc.Button("Table Description", id="Table-Description-B", n_clicks=0, color="info"),
                 width={"size": "auto", "offset": 1})]),
    dbc.Modal([
    dbc.ModalHeader("Table Description"),
    dbc.ModalBody(t_description,style={"color": "blue"}),
    dbc.ModalFooter(dbc.Button("Close", id="Table-Description-close", n_clicks=0))], id="Table-Description"),
    dbc.Row([dbc.Col(
    dash_table.DataTable(id="table-6",
                            columns=[],
                            data=[],
                            row_selectable="single",
                            selected_rows=[],
                            style_header={"border": "2px solid blue", "color": "black", "background": "#f1f1c1"},
                            style_data={"border": "2px solid blue", "color": "black", "background": "white"}),
            width={"size": "auto", "offset": 1})]),
    html.Br(),
    ],id="hide-2",style={"background": "grey", "border": "5px blue solid", "overflowX": 'auto', 'display': 'none'}),

    html.Div([
    dbc.Row([dbc.Col(dbc.Alert("The Chosen Facilities Are" , color="primary"), width=5),]),
    dbc.Row([dbc.Col(html.Div([],id="f-list"), width={"size": "auto", "offset": 1}),]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("The Total Prod/Trans  Cost", color="primary"), width=5), ]),
    dbc.Row([dbc.Col(html.Div([], id="result-1"), width={"size": "auto", "offset": 1}), ]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("The Total Operation  Cost", color="primary"), width=5), ]),
    dbc.Row([dbc.Col(html.Div([], id="result-2"), width={"size": "auto", "offset": 1}), ]),
    html.Hr(),
    dbc.Row([dbc.Col(dbc.Alert("The Total  Cost" , color="primary"), width=5),]),
    dbc.Row([dbc.Col(html.Div([],id="result-3"), width={"size": "auto", "offset": 1}),]),
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
    ])],id="hide-3",style={"background":"grey","border":"5px blue solid","overflowX": 'auto','display': 'none'}),


    html.Div([
    dbc.Row([dbc.Col(dbc.Alert("Shipment Vs Capacity", color="primary"), width=5)]),
    dbc.Row([dbc.Col(html.Div([], id="Shipment_Vs_Capacity"), width={"size": "auto", "offset": 1})]),
    html.Hr(),
    dcc.Graph(id="graph-1"),
    dbc.Row([dbc.Col(dcc.Slider(id="slider-1",min=0,step=None,marks={},value=0),width={"size": 10, "offset": 1})]),
    html.Br()
    ],id="hide-4",style={"background":"grey","border":"5px blue solid","overflowX": 'auto','display': 'none'}),

    html.Div([
    dbc.Row([dbc.Col(dbc.Alert("Shipment Vs Demand" , color="primary"), width=5)]),
    dbc.Row([dbc.Col(html.Div([], id="Shipment_Vs_Demand"), width={"size": "auto", "offset": 1})]),
    html.Hr(),
    dcc.Graph(id="graph-2"),
    dbc.Row([dbc.Col(dcc.Slider(id="slider-2", min=0, step=None, marks={}, value=0), width={"size": 10, "offset": 1})]),
    html.Br()
    ], id="hide-5", style={"background": "grey", "border": "5px blue solid", "overflowX": 'auto', 'display': 'none'}),

    html.Div([html.Br() for i in range(0,10)])

    ])
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([rows])

@app.callback(
     dash.dependencies.Output("Modal-Description", "is_open"),
     [dash.dependencies.Input("Model-Description-B", "n_clicks"),
      dash.dependencies.Input("Modal-Description-close", "n_clicks")],
      [dash.dependencies.State("Modal-Description", "is_open")]
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
    op_list.append(problem.df_fixed_cost.to_dict("records"))
    op_list.append([{"name": i, "id": i} for i in problem.df_fixed_cost.columns])
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
    op_list.append(problem.no_of_w)
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
    op_list.append([])
    op_list.append([])
    op_list.append(style0)
    op_list.append(style1)
    op_list.append(is_open)
    op_list.append(" ")
    op_list.append(problem.no_of_f)
    op_list.append(problem.no_of_w)
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
     dash.dependencies.Output("table-5", "data"),
     dash.dependencies.Output("table-5", "columns"),
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
     dash.dependencies.Input("close-modal-0", "n_clicks")],
     [dash.dependencies.State("dim-1", "value"),
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
            problem.no_of_w = int(v2)
            problem.no_of_p = int(v3)
            problem.index_headers = ["Factories", "Products"]
            string1 = "Warehouse-"
            string2 = "Factory-"
            problem.facility_names.clear()
            for i in range(0, problem.no_of_f):
                problem.facility_names.append(string2 + str(i + 1))
            problem.warehouse_names.clear()
            for i in range(0, problem.no_of_w):
                problem.warehouse_names.append(string1 + str(i + 1))
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
            problem.no_of_w = int(v2)
            problem.index_headers = ["Factories", "Products"]
            string1 = "Warehouse-"
            string2 = "Factory-"
            problem.facility_names.clear()
            for i in range(0, problem.no_of_f):
                problem.facility_names.append(string2 + str(i + 1))
            problem.warehouse_names.clear()
            for i in range(0, problem.no_of_w):
                problem.warehouse_names.append(string1 + str(i + 1))
            return show_zero_hide_one(is_open)
    elif "upload" in changed_id:
        temp =[False]*5
        for i in list_of_names:
            if "xls" in i:
                temp.append(True)
            else:
                temp.append(False)
            if "shipment_cost" in i:
                temp[0] = True
            elif "fixed_operation_cost" in i:
                temp[1] = True
            elif "production_cost" in i:
                temp[2] = True
            elif "demand" in i:
                temp[3] = True
            elif "capacity" in i:
                temp[4] = True
        mask = np.array(temp)
        mask = mask.all()
        if len(list_of_contents) == 5 and mask:
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
                    problem.warehouse_names = df_shipmen_cost.columns.values.tolist()
                    problem.no_of_w = len(problem.warehouse_names)
                    problem.cost_for_f_to_wh = np.array(df_shipmen_cost).flatten()
                elif "fixed_operation_cost" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_operation_cost = pd.read_excel(io.BytesIO(decoded))
                    headers = df_operation_cost.columns.values.tolist()
                    df_operation_cost.drop(headers[0], axis=1, inplace=True)
                    problem.fixed_operating_cost = np.array(df_operation_cost).flatten()
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
                    problem.w_demand_constraint = np.array(df_demand).flatten()
                elif "capacity" in i:
                    content_type, content_string = j.split(',')
                    decoded = base64.b64decode(content_string)
                    df_capacity = pd.read_excel(io.BytesIO(decoded))
                    headers = df_capacity.columns.values.tolist()
                    df_capacity.drop([headers[0], headers[1]], axis=1, inplace=True)
                    problem.f_capacity_constraint = np.array(df_capacity).flatten()
            return hide_zero_show_one(is_open)
        else:
            msg = """Five excel files should be selected.Names of excel files should be like 1-shipment_cost, 
                  2-fixed_operation_cost, 3-production_cost, 4-capacity and 5-demand"""
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
    [dash.dependencies.Output("table-6", "data"),
     dash.dependencies.Output("table-6", "columns"),
     dash.dependencies.Output("hide-2", "style"),
     dash.dependencies.Output("modal-1", "is_open"),
     dash.dependencies.Output("modal-1-content", "children"),
     dash.dependencies.Output("scale", "value")],
    [dash.dependencies.Input("solve",  "n_clicks"),
    dash.dependencies.Input("close-modal-1", "n_clicks")],
    [dash.dependencies.State("table-1", "data"),
    dash.dependencies.State("table-2", "data"),
    dash.dependencies.State("table-3", "data"),
    dash.dependencies.State("table-4", "data"),
    dash.dependencies.State("table-5", "data"),
    dash.dependencies.State("scale", "value"),
    dash.dependencies.State("modal-1", "is_open")
     ])
def solve_model(n1,n2,shipment_cost,oper_cost,prod_cost,demand,capa,u,is_open):
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
        df1.drop([headers[0], headers[1]], axis=1, inplace=True)
        problem.cost_for_f_to_wh = np.array(df1).flatten().astype(np.float)
        df2 = pd.DataFrame(oper_cost)
        headers = df2.columns.values.tolist()
        df2.drop(headers[0], axis=1, inplace=True)
        problem.fixed_operating_cost = np.array(df2).flatten().astype(np.float)
        df3 = pd.DataFrame(prod_cost)
        headers = df3.columns.values.tolist()
        df3.drop([headers[0], headers[1]], axis=1, inplace=True)
        problem.cost_for_production = np.array(df3).flatten().astype(np.float)
        df4 = pd.DataFrame(demand)
        headers = df4.columns.values.tolist()
        df4.drop(headers[0], axis=1, inplace=True)
        df4 = df4.transpose()
        problem.w_demand_constraint = np.array(df4).flatten().astype(np.float)
        df5 = pd.DataFrame(capa)
        headers = df5.columns.values.tolist()
        df5.drop([headers[0], headers[1]], axis=1, inplace=True)
        problem.f_capacity_constraint = np.array(df5).flatten().astype(np.float)

        mask1 = np.isnan(problem.cost_for_f_to_wh)
        mask1 = np.invert(mask1).all()
        mask2 = np.isnan(problem.fixed_operating_cost)
        mask2 = np.invert(mask2).all()
        mask3 = np.isnan(problem.cost_for_production)
        mask3 = np.invert(mask3).all()
        mask4 = np.isnan(problem.w_demand_constraint)
        mask4 = np.invert(mask4).all()
        mask5 = np.isnan(problem.f_capacity_constraint)
        mask5 = np.invert(mask5).all()
        data = []
        column = []
        if mask1 and mask2 and mask3 and mask4 and mask5:
            if not is_open:
                problem.solve_problem()
            if problem.result_status == "success":
                data = problem.df_combinations.to_dict("records")
                column =  [{"name": i, "id": i} for i in problem.df_combinations]

                style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'block'}
            else:
                style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'none'}

            return data, column, style, not is_open, problem.solution_message,u
        else:
            style = {"background": "grey", "border": "2px blue solid","overflowX": 'auto', 'display': 'none'}
            return data, column, style, not is_open,"Empty Data Fields Are Found",u



@app.callback(
    dash.dependencies.Output("Table-Description", "is_open"),
    [dash.dependencies.Input("Table-Description-B", "n_clicks"),
     dash.dependencies.Input("Table-Description-close", "n_clicks")],
     [dash.dependencies.State("Table-Description", "is_open")]
)
def table_description(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [dash.dependencies.Output("hide-3", "style"),
     dash.dependencies.Output("f-list", "children"),
     dash.dependencies.Output("result-1", "children"),
     dash.dependencies.Output("result-2", "children"),
     dash.dependencies.Output("result-3", "children"),
     dash.dependencies.Output("result-shipment", "children"),
     dash.dependencies.Output("slider-1", "value"),
     dash.dependencies.Output("slider-2", "value")],
    [dash.dependencies.Input("table-6", "selected_rows")])
def show_comb_result(row_no):
    style = {"background": "grey", "border": "2px blue solid", "overflowX": 'auto', 'display': 'block'}
    if row_no :
        problem.make_individual_results(row_no[0])
        n = problem.combination_result_set[3][row_no[0]]
        string3 = ""
        string1 = ""
        for i in problem.combination_solve_set[2][n]:
            string1 = string1 + string3 + i
            string3 = ","
        v1 = problem.combination_result_set[0][problem.result_pointer]
        v2 = problem.combination_result_set[1][problem.result_pointer]
        v3 = problem.combination_result_set[2][problem.result_pointer]
        table = generate_m_table(problem.df_shipment)
        return style, html.H5(string1), html.H5( str(v1)), html.H5(str(v2)), html.H5(str(v3)), table, 0, 0
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output("hide-4", "style"),
     dash.dependencies.Output("hide-5", "style"),
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
        comb_pos = problem.combination_result_set[3][problem.result_pointer]
        f_list = problem.combination_solve_set[1][comb_pos].copy()
        size = len(f_list)
        g_title = "Shipment Vs Capacity For " + f_list[v1]
        table = generate_m_table(problem.df_capacity_slack)

        df = problem.df_capacity_slack.loc[f_list[v1]]
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
        marks = {i:{ "label":f_list[i],"style":{"color":"blue"}} for i in range(0,size)}
        return style_active, style_hide, table, html.Div(), figure, {}, size, marks, problem.no_of_w,{}
    elif "constraint-2" in changed_id or "slider-2" in changed_id:
        comb_pos = problem.combination_result_set[3][problem.result_pointer]
        f_list = problem.combination_solve_set[1][comb_pos].copy()
        size = len(f_list)
        g_title = "Shipment Vs Demand For " + problem.warehouse_names[v2]
        table = generate_m_table(problem.df_demand_slack)
        df = problem.df_demand_slack_graph.loc[problem.warehouse_names[v2]]
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
        marks = {i: {"label": problem.warehouse_names[i], "style": {"color": "blue"}} for i in range(0, problem.no_of_w)}
        return style_hide, style_active, html.Div(), table, {}, figure, size, {}, problem.no_of_w, marks
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(port=8086,debug=True)

