import pandas as pd

def output_to_excel(data, metric_name, output_name):
    df = pd.DataFrame({metric_name: data})

    mean = df[metric_name].mean()
    std = df[metric_name].std()

    # create Excel and write data
    writer = pd.ExcelWriter(output_name, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Data')
    worksheet = writer.sheets['Data']
    worksheet.cell(row=len(data) + 2, column=1, value='Mean')
    worksheet.cell(row=len(data) + 2, column=2, value=mean)
    worksheet.cell(row=len(data) + 3, column=1, value='Std')
    worksheet.cell(row=len(data) + 3, column=2, value=std)
    writer.save()

if __name__ == '__main__':
    data = [1, 2, 3, 4, 5, 7, 8, 9]
    output_name = 'output_name.xlsx'

    output_to_excel(data, 'AUC', output_name)
