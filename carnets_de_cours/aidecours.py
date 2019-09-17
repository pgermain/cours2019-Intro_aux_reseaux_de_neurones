from matplotlib import pyplot as plt
import numpy as np


def set_current_axes(ax):
    if ax is not None:
        plt.sca(ax)

def show_1d_function(fct, min_val=-5, max_val=5, step=.1, constante_x=None, constante_y=None, ax=None, **kwargs):
    x_values = np.linspace(min_val, max_val, int((max_val-min_val)/step)+1)
    fct_values = fct(x_values)
    
    set_current_axes(ax)
    plt.plot(x_values, fct_values, **kwargs)
    
    if 'label' in kwargs:
        plt.legend()
        
    if constante_x is not None:
        plt.plot([min_val, max_val], [constante_x,constante_x], 'k--')
        
    if constante_y is not None:
        plt.plot([constante_y, constante_y], [min(fct_values),max(fct_values)], 'k--')    

def show_2d_function(fct, min_val=-5, max_val=5, mesh_step=.01, optimal=None, bar=True, ax=None, **kwargs):
    x1_min, x2_min = np.ones(2) * min_val
    x1_max, x2_max = np.ones(2) * max_val
    
    x1_values = np.arange(x1_min, x1_max+mesh_step, mesh_step)
    x2_values = np.arange(x2_min, x2_max+mesh_step, mesh_step)
  
    fct_values = np.array([[fct(np.array((x1,x2))) for x1 in x1_values] for x2 in x2_values])
    
    set_current_axes(ax)
    if 'cmap' not in kwargs: kwargs['cmap'] = 'RdBu'
    plt.contour(x1_values, x2_values, fct_values, 40, **kwargs)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    
    if bar:
        plt.colorbar()
        
    if optimal is not None: 
        plt.scatter(*optimal, s=200, marker='*', c='r')

def show_2d_dataset(x, y, ax=None):
    set_current_axes(ax)
    labels = np.unique(y)
    for one_label, one_color in zip(labels, "rby"):
        mask = (y==one_label)
        plt.scatter(x[mask, 0], x[mask, 1], c=one_color, edgecolors='k', label=f'$y={one_label}$')

    plt.legend()
    
def show_2d_predictions(x, y, predict_fct, ax=None):
    # Inspired by: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    mesh_step = .02  # step size in the mesh
    x0_min, x0_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    x1_min, x1_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    
    set_current_axes(ax)
    if predict_fct is not None:
        x0_values = np.arange(x0_min, x0_max, mesh_step)
        x1_values = np.arange(x1_min, x1_max, mesh_step)
        x0_grid, x1_grid = np.meshgrid(x0_values, x1_values)
        x_grid_pairs = np.c_[x0_grid.ravel(), x1_grid.ravel()]
        grid_predictions = predict_fct(x_grid_pairs)
        
        # Put the result into a color plot
        grid_predictions = grid_predictions.reshape(x0_grid.shape)
        plt.contourf(x0_grid, x1_grid, grid_predictions, cmap='RdBu', alpha=.8)
            
    # Plot also the training points
    show_2d_dataset(x, y)
    return x0_min, x0_max, x1_min, x1_max

def show_2d_vector_field(grad_fct, min_val=-5, max_val=5, mesh_step=.5, ax=None, **kwargs):
    x1_min, x2_min = np.ones(2) * min_val
    x1_max, x2_max = np.ones(2) * max_val
    
    x1_values = np.arange(x1_min, x1_max+mesh_step, mesh_step)
    x2_values = np.arange(x2_min, x2_max+mesh_step, mesh_step)
  
    grad_values = np.array([[grad_fct(np.array((x1,x2))) for x1 in x1_values] for x2 in x2_values])

    set_current_axes(ax)
    plt.quiver(x1_values, x2_values, grad_values[:,:,0], grad_values[:,:,1], **kwargs)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))

# Graphique de la trajectoire de descente en gradient
def show_2d_trajectory(w_list, fct, min_val=-5, max_val=5, mesh_step=.5, w_opt=None, ax=None):
    show_2d_function(fct, min_val, max_val, mesh_step, optimal=w_opt, ax=ax) 
    
    if len(w_list) > 0:
        trajectory = np.array(w_list)
        plt.plot(trajectory[:,0], trajectory[:,1], 'o--', c='g')
    
    plt.title('Trajectoire de la descente en gradient'); plt.xlabel('$w_1$'); plt.ylabel('$w_2$')

# Graphique des valeurs de la fonction objectif lors de l'apprentissage
def show_learning_curve(obj_list, obj_opt=None, ax=None):
    set_current_axes(ax)
    plt.plot(obj_list, 'o--', c='g', label='$F(\mathbf{w})$')
    if obj_opt is not None: plt.plot([0, len(obj_list)], 2*[obj_opt], '*--', c='r', label='optimal');
    plt.title('Valeurs de la fonction objectif'); plt.xlabel('$t$')
    plt.legend()
    
def code_button():
    from IPython.display import HTML
    return HTML('''<script>
    code_show=false; 
    function code_toggle() {
    if (code_show){
         $('div.input').hide();
    } else {
         $('div.input').show();
    }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    <a href="javascript:code_toggle()">voir/cacher le code</a>.''')

def center_images():
    from IPython.display import display, HTML
    display(HTML("""
    <style>
       .output_png {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
    </style>
    """))


if __name__ == '__main__':
    show_2d_function(lambda x: x @ x)
    plt.show()



