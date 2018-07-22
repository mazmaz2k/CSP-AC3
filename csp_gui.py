import tkinter as tk
from PIL import Image, ImageTk
import csp


class CspGui:


    def run_algorithm(self):
        print(self.num_of_vertices.get())
        print(self.num_of_edges.get())
        print(self.num_of_colors.get())
        print(self.var.get())
        print(self.max_num_of_edges.get())
        # print('00000000000000000000000000000000000000000000000000000000000000')






        M, g = csp.create_graph(self.num_of_vertices.get(), self.num_of_edges.get() , self.max_num_of_edges.get())
        graph = csp.Graph(g)
        colors = csp.colors_for_map(self.num_of_colors.get())
        print(graph)
        print("Colors: ", colors)
        constraints = []
        for edge in graph.edges():
            if len(edge) is not 2:
                continue
            le = list(edge)
            # print(le, " ",edge)
            c = csp.BinaryConstraint(le[0], le[1])
            constraints.append(c)
        csp_var = csp.ConstraintSatisfactionProblem(list(graph.vertices()), colors, constraints)
        csp.print_graph_gui(M, g, csp.solve(csp_var), len(colors))
        print(csp.solve(csp_var))














        img = Image.open('graph.jpg')

        '''# Change size of image if needed.
        width, height = img.size

        if width > self.width/2-10:
            width = self.width/2-10

        if height > self.height/6*5-40:
            height = self.height/6*5-40

        img = img.resize((int(width), int(height)))'''
        photo = ImageTk.PhotoImage(img)


        # Delete previous label which has image.
        for i in self.image_frame.winfo_children():
            if i.winfo_class() == 'Label':
                i.destroy()

        # Create new label with the selected image.
        label = tk.Label(
            self.image_frame,
            #bg='gray80',
            image=photo,
            bd=0
        )
        label.pack()
        label.photo = photo

        print('\n\ndone\n\n')










    def __init__(self):
        
        self.root = tk.Tk()
        self.root.state('zoomed') #Open in fullscreen.

        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()


        tk.Label(self.root).grid(row=0, padx=100)
        tk.Label(self.root,  text='Number of vertices: ', font=14).grid(row=0, column=1, sticky='w', padx=30)
        self.num_of_vertices = tk.Scale(self.root, from_=1, to=100, orient='horizontal', length=150)
        self.num_of_vertices.grid(row=0, column=2, sticky='w')

        tk.Label(self.root).grid(row=0, padx=100)
        tk.Label(self.root, text='Number of edges: ', font=14).grid(row=1, column=1, sticky='w', padx=40)
        self.num_of_edges = tk.Scale(self.root, from_=1, to=50, orient='horizontal')
        self.num_of_edges.grid(row=1, column=2, sticky='w')

        tk.Label(self.root).grid(row=0, padx=100)
        tk.Label(self.root, text='Max number of edges: ', font=14).grid(row=1, column=0, sticky='w', padx=30)
        self.max_num_of_edges = tk.Scale(self.root, from_=20, to=500, resolution=10,orient='horizontal')
        self.max_num_of_edges.grid(row=2, column=0, sticky='w')



        tk.Label(self.root).grid(row=0, padx=100)
        tk.Label(self.root, text='Number of colors: ', font=14).grid(row=2, column=1, sticky='w', padx=30)
        self.num_of_colors = tk.Scale(self.root, from_=1, to=10, orient='horizontal')
        self.num_of_colors.grid(row=2, column=2, sticky='w')

        self.run = tk.Button(self.root, text='Run Algorithm', font=18)
        self.run.grid(row=0, column=3, rowspan=3, sticky='w', padx=200)

        self.var = tk.IntVar()
        self.arc_consistency = tk.Checkbutton(self.root, text='Use arc consistency', font=14, variable=self.var)
        self.arc_consistency.grid(row=2, column=3, sticky='w', padx=200)

        self.run.config(command=lambda: self.run_algorithm())

        self.image_frame = tk.Frame(
            self.root,
            bg='white',
            width=10,
            height=100
        )
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.image_frame.grid(row=3, column=0, sticky='news', columnspan=4)
        #self.image_frame.pack_propagate(False)
        #self.image_frame.pack(expand='true', fill='both')

        

        self.root.title('CSP Map Coloring')
        self.root.mainloop()





win = CspGui()