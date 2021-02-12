import sys

# Write output to file
#f = open("duplicate_out.txt", 'w')
#sys.stdout = f

# Existing chemical network
file1 = "Stand-2019December"
# Chemical network to be added in
file2 = "Stand-2019Venus"
# New chemical network to write to
file3 = "Stand-2019Venus_Sean_Dec"


# Initialise reactant and product variables as empty strings
f1_r1=""
f1_r2=""
f1_p1=""
f1_p2=""
f1_p3=""
f2_r1=""
f2_r2=""
f2_p1=""
f2_p2=""
f2_p3=""

d=0
j=0
l=0
duplicate = False
emptyline=True

with open(file2) as f2:
    for f2line in f2:
#        print(f2line)
        #Test what the code is doing then break
        #if l==30:
        #    break
        k=0
        l+=1    
        j=0
        duplicate=False
        emptyline=True
        
        # Append characters to reactant and product variables
        for char in f2line:
            # First reactant
            if char != " " and k < 8:
                f2_r1 += char
                emptyline=False
            # Second reactant if there is one
            elif char != ' ' and k >= 8 and k < 24:
                f2_r2 += char
            # First product
            elif char != ' ' and k >= 24 and k < 32:
                f2_p1 += char
            # Second product if there is one
            elif char != ' ' and k >= 32 and k < 40:
                f2_p2 += char
            # Third product if there is one
            elif char != ' ' and k >= 40 and k < 48:
                f2_p3 += char
            # Count no. of columns read in the line
            k=k+1

        # Print the reactions
#        print(f2_r1 + " + " + f2_r2 + " = " + f2_p1 + " + " + f2_p2 + " + " + f2_p3)
#        print(l)                    

        with open(file1) as f1:
            for f1line in f1:
#                print(f1line)
                #if j==100:
                #    break
                
                i=0
                j+=1

                # Append characters to reactant and product variables
                for ch in f1line:
                    # First reactant
                    if ch != " " and i < 8:
                        f1_r1 += ch
                    # Second reactant if there is one
                    elif ch != ' ' and i >= 8 and i < 24:
                            f1_r2 += ch
                    # First product
                    elif ch != ' ' and i >= 24 and i < 32:
                        f1_p1 += ch
                    # Second product if there is one
                    elif ch != ' ' and i >= 32 and i < 40:
                        f1_p2 += ch
                    # Third product if there is one
                    elif ch != ' ' and i >= 40 and i < 48:
                        f1_p3 += ch
                    # Count no. of columns read in the line
                    i=i+1
                        
                
                # Cross check reactants and products for every line of f1 for a given line of f2
                #
                # Check if same reactants
                if f1_r1==f2_r1 and f1_r2==f2_r2:
                    # Check if same products in any order
                    if f1_p1==f2_p1 and f1_p2==f2_p2 and f1_p3==f2_p3:
                        duplicate=True
                    elif f1_p1==f2_p2 and f1_p2==f2_p1 and f1_p3==f2_p3:
                        duplicate=True
                    elif f1_p1==f2_p3 and f1_p2==f2_p2 and f1_p3==f2_p1:
                        duplicate=True
                    elif f1_p1==f2_p1 and f1_p2==f2_p3 and f1_p3==f2_p2:
                        duplicate=True
                    elif f1_p1==f2_p2 and f1_p2==f2_p3 and f1_p3==f2_p1:
                        duplicate=True
                    elif f1_p1==f2_p3 and f1_p2==f2_p1 and f1_p3==f2_p2:
                        duplicate=True

                # Reactants could be listed the other way around and still be the same    
                elif f1_r1==f2_r2 and f1_r2==f2_r1:
                    # Check if same products in any order
                    if f1_p1==f2_p1 and f1_p2==f2_p2 and f1_p3==f2_p3:
                        duplicate=True
                    elif f1_p1==f2_p2 and f1_p2==f2_p1 and f1_p3==f2_p3:
                        duplicate=True
                    elif f1_p1==f2_p3 and f1_p2==f2_p2 and f1_p3==f2_p1:
                        duplicate=True
                    elif f1_p1==f2_p1 and f1_p2==f2_p3 and f1_p3==f2_p2:
                        duplicate=True
                    elif f1_p1==f2_p2 and f1_p2==f2_p3 and f1_p3==f2_p1:
                        duplicate=True
                    elif f1_p1==f2_p3 and f1_p2==f2_p1 and f1_p3==f2_p2:
                        duplicate=True

                if duplicate == True:
                    print("\n\nThis reaction is in both files\n")# at line "+str(l)+" in new file and line "+str(j)+" in the original file.")
                    print("line "+str(l)+" in "+file2+":     "+f2line)
                    print("line "+str(j)+" in "+file1+": "+f1line)
                    d+=1
                    break

#                print(f1_r1 + " + " + f1_r2 + " = " + f1_p1 + " + " + f1_p2 + " + " + f1_p3)
#                print(j)

                # Empty the reactant and product variables to be appended again
                f1_r1=""
                f1_r2=""
                f1_p1=""
                f1_p2=""
                f1_p3=""

        # Empty the reactant and product variables to be appended again
        f2_r1=""
        f2_r2=""
        f2_p1=""
        f2_p2=""
        f2_p3=""

        # If not a duplicated reaction add it to the network
        if duplicate==False and emptyline==False:
            with open(file3, "a") as myfile:
                myfile.write(f2line)

# Print no. of duplicated reactions
print("The no. of duplicated reactions is "+str(d))   


#f.close()
