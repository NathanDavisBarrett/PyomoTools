import pyomo.kernel as pmo

def ModelToDict(model:pmo.block):
    dct = {}
    for c in model.children():
        cName = c.local_name
        if isinstance(c,(pmo.variable_list,pmo.variable_tuple)):
            dct[cName] = [pmo.value(c[i],exception=False) for i in range(len(c))]
        elif isinstance(c,pmo.variable_dict):
            dct[cName] = {k:pmo.value(c[k],exception=False) for k in c}
        elif isinstance(c,pmo.variable):
            dct[cName] = pmo.value(c,exception=False)
        
        elif isinstance(c,(pmo.block_list,pmo.block_tuple)):
            dct[cName] = [ModelToDict(c[i]) for i in range(len(c))]
        elif isinstance(c,pmo.block_dict):
            dct[cName] = {k:ModelToDict(c[k]) for k in c}
        elif isinstance(c,pmo.block):
            dct[cName] = ModelToDict(c)
    return dct