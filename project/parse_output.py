import pandas as pd
import os


def parse_output(output, buildings, buildings_constructed, logging, folder_output):
    logging.debug('Parsing output')
    output['Stock segmented'] = pd.DataFrame(buildings._stock_seg_dict)
    output['Stock knowledge energy performance'] = pd.DataFrame(buildings._stock_knowledge_ep_dict)
    output['Stock construction segmented'] = pd.DataFrame(buildings_constructed._stock_constructed_seg_dict)
    output['Stock knowledge construction'] = pd.DataFrame(buildings_constructed._stock_knowledge_construction_dict)

    keys = ['Stock segmented', 'Stock knowledge energy performance', 'Stock construction segmented',
            'Stock knowledge construction']
    for k in keys:
        name_file = os.path.join(folder_output, '{}.csv'.format(k.lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        output[k].to_csv(name_file, header=True)
        name_file = os.path.join(folder_output, '{}.pkl'.format(k.lower().replace(' ', '_')))
        output[k].to_pickle(name_file)

    def to_grouped(df, level):
        grouped_sum = df.groupby(level).sum()
        summed = grouped_sum.sum()
        summed.name = 'Total'
        result = pd.concat((grouped_sum.T, summed), axis=1).T
        return result

    levels = ['Energy performance', 'Heating energy']
    val = 'Stock segmented'
    for lvl in levels:
        name_file = os.path.join(folder_output, '{}.csv'.format((val + '_' + lvl).lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        to_grouped(output[val], lvl).to_csv(name_file, header=True)

    keys = ['Population total', 'Population', 'Population housing', 'Flow needed']
    name_file = os.path.join(folder_output, 'demography.csv')
    temp = pd.concat([output[k] for k in keys], axis=1)
    temp.columns = keys
    temp.to_csv(name_file, header=True)
    name_file = os.path.join(folder_output, 'demography.pkl')
    temp.to_pickle(name_file)

    new_output = {}
    for key, output_dict in output.items():
        if isinstance(output_dict, dict):
            if isinstance(val, pd.DataFrame):
                new_item = {yr: itm.stack(itm.columns.names) for yr, itm in output_dict.items()}
                new_output[key] = pd.DataFrame(new_item)
            elif isinstance(val, pd.Series):
                new_output[key] = pd.DataFrame(output_dict)
            elif isinstance(val, float):
                new_output[key] = pd.Series(output[key])
    for key in new_output.keys():
        name_file = os.path.join(folder_output, '{}.csv'.format(key.lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        new_output[key].to_csv(name_file, header=True)
        name_file = os.path.join(folder_output, '{}.pkl'.format(key.lower().replace(' ', '_')))
        new_output[key].to_pickle(name_file)