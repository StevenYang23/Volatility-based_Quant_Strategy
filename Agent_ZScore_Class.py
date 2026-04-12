class Agent_ZScore:
    def __init__(self, k = 0.65):
        self.port_value = 0
        self.prev_port_value = 0
        self.num_options = 0 # number of options to trade (1 or 0)
        self.num_underlying = 0 # number of underlying shares to trade
        self.k = k # z-score threshold
        self.vrp_list = []
        self.delta_list = []
        self.gamma_list = []
        self.vega_list = []
        self.theta_list = []
        self.vanna_list = []
        self.volga_list = []
        self.rho_list = []

    def trade(self, data):
        self.prev_port_value = self.port_value
        self.vrp_list.append(data["Straddle_imp_vol"] - data["RV"])
        if data['Force_Close']:
            
        
