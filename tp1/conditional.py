import csv


RANK_VALUE = 2  # 1, 2, 3, 4
GRE_VALUE = 1   # 0, 1
GPA_VALUE = 1  # 0, 1

with open('binary.csv') as binary:
    reader = csv.reader(binary)
    next(reader)   # Saltear el encabezado del csv
    
    # Counters
    count_gre = 0
    count_gpa = 0
    count_intersection = 0
    rank_count = 0
    for row in reader:
        # Parsing
        fact = {}
        fact["admit"] = int(row[0])
        fact["gre"] = 0 if float(row[1]) >= 500 else 1  # Discretizacion
        fact["gpa"] = 0 if float(row[2]) >= 3 else 1    # Discretizacion
        fact["rank"] = int(row[3])

        # Conteo
        if fact["rank"] == RANK_VALUE: # Condicionado a rank = RANK_VALUE
            rank_count += 1
            if fact["gre"] == GRE_VALUE:
                count_gre +=1

            if fact["gpa"] == GPA_VALUE:
                count_gpa += 1
            
            if fact["gpa"] == GRE_VALUE and fact["gre"] == GPA_VALUE:
                count_intersection += 1

        
# Deberian ser iguales por la independencia condicional pero no lo son
print(count_intersection/rank_count)                  # P(GRE = 1, GPA = 1 | Rank = 1)
print((count_gpa/rank_count) * (count_gre/rank_count))  # P(GRE = 1 | Rank = 1) * P(GPA = 1 | Rank = 1 )

#print((count_gpa/rank_count))
#print((count_gre/rank_count))
