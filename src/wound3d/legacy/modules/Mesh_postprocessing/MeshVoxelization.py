class VoxelMesh:
    """
    A class for voxelizing 3D meshes and computing voxel-level properties.
    
    This class converts a mesh into a voxel grid and computes various statistics
    for each voxel including:
    - Average/median normal vectors
    - Normal variance (measure of surface smoothness)
    - Average color and color variance
    - Curvature proxy (based on normal variation)
    """
    
    def __init__(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
                 mesh_vertex_colors: Optional[np.ndarray] = None,
                 voxel_size: Optional[float] = None,
                 num_voxels: Optional[int] = None):
        """
        Initialize the VoxelMesh.
        
        Args:
            mesh_vertices: Array of shape (N, 3) containing vertex coordinates
            mesh_faces: Array of shape (M, 3) containing face indices
            mesh_vertex_colors: Array of shape (N, 3) or (N, 4) containing RGB or RGBA colors
            voxel_size: Size of each voxel (in mesh units). If None, calculated from num_voxels
            num_voxels: Approximate number of voxels along longest axis. Default is 50
        """
        self.vertices = mesh_vertices
        self.faces = mesh_faces
        self.vertex_colors = mesh_vertex_colors
        
        # Create trimesh object
        self.mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            process=False
        )
        
        # Calculate vertex normals
        self.vertex_normals = self.mesh.vertex_normals
        
        # Determine voxel size
        if voxel_size is None:
            if num_voxels is None:
                num_voxels = 50
            mesh_extent = self.mesh.extents.max()
            self.voxel_size = mesh_extent / num_voxels
        else:
            self.voxel_size = voxel_size
        
        # Initialize voxel properties
        self.voxel_grid = None
        self.voxel_centers = None
        self.voxel_properties = {}
        
        logger.info(f"Initialized VoxelMesh with {len(mesh_vertices)} vertices, "
                   f"{len(mesh_faces)} faces, voxel_size={self.voxel_size:.4f}")
    
    def voxelize(self):
        """
        Voxelize the mesh and compute voxel indices for each vertex.
        """
        logger.info("Voxelizing mesh...")
        
        # Get mesh bounds
        min_bound = self.vertices.min(axis=0)
        max_bound = self.vertices.max(axis=0)
        
        # Calculate grid dimensions
        grid_dims = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int) + 1
        
        # Assign vertices to voxels
        voxel_indices = np.floor((self.vertices - min_bound) / self.voxel_size).astype(int)
        
        # Create unique voxel identifiers
        self.voxel_grid = {}
        for i, voxel_idx in enumerate(voxel_indices):
            voxel_key = tuple(voxel_idx)
            if voxel_key not in self.voxel_grid:
                self.voxel_grid[voxel_key] = {
                    'vertex_indices': [],
                    'normals': [],
                    'colors': []
                }
            self.voxel_grid[voxel_key]['vertex_indices'].append(i)
            self.voxel_grid[voxel_key]['normals'].append(self.vertex_normals[i])
            if self.vertex_colors is not None:
                self.voxel_grid[voxel_key]['colors'].append(self.vertex_colors[i])
        
        # Calculate voxel centers
        self.voxel_centers = {}
        for voxel_key in self.voxel_grid:
            voxel_coords = np.array(voxel_key)
            center = min_bound + (voxel_coords + 0.5) * self.voxel_size
            self.voxel_centers[voxel_key] = center
        
        logger.info(f"Created {len(self.voxel_grid)} voxels")
    
    def compute_voxel_properties(self):
        """
        Compute statistical properties for each voxel.
        
        Properties computed:
        - average_normal: Mean normal vector
        - median_normal: Median normal vector
        - normal_variance: Variance of normals (smoothness measure)
        - average_color: Mean color
        - color_variance: Variance of color
        - curvature_proxy: Measure of local curvature based on normal variation
        """
        if self.voxel_grid is None:
            self.voxelize()
        
        logger.info("Computing voxel properties...")
        
        for voxel_key, voxel_data in self.voxel_grid.items():
            normals = np.array(voxel_data['normals'])
            
            # Compute normal statistics
            avg_normal = np.mean(normals, axis=0)
            avg_normal_normalized = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
            
            median_normal = np.median(normals, axis=0)
            median_normal_normalized = median_normal / (np.linalg.norm(median_normal) + 1e-8)
            
            # Normal variance (measure of how much normals vary in this voxel)
            normal_variance = np.var(normals, axis=0).mean()
            
            # Curvature proxy: measure angular deviation from mean normal
            # Higher values indicate more curvature
            dot_products = np.dot(normals, avg_normal_normalized)
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angles = np.arccos(dot_products)
            curvature_proxy = np.std(angles)
            
            # Compute color statistics if available
            avg_color = None
            color_variance = None
            if voxel_data['colors']:
                colors = np.array(voxel_data['colors'])
                avg_color = np.mean(colors, axis=0)
                color_variance = np.var(colors, axis=0).mean()
            
            # Store properties
            self.voxel_properties[voxel_key] = {
                'average_normal': avg_normal_normalized,
                'median_normal': median_normal_normalized,
                'normal_variance': normal_variance,
                'average_color': avg_color,
                'color_variance': color_variance,
                'curvature_proxy': curvature_proxy,
                'num_vertices': len(voxel_data['vertex_indices'])
            }
        
        logger.info(f"Computed properties for {len(self.voxel_properties)} voxels")
    
    def get_voxel_property_arrays(self):
        """
        Get voxel properties as numpy arrays for easy analysis.
        
        Returns:
            Dictionary containing arrays of voxel properties
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        num_voxels = len(self.voxel_properties)
        centers = np.zeros((num_voxels, 3))
        avg_normals = np.zeros((num_voxels, 3))
        median_normals = np.zeros((num_voxels, 3))
        normal_variances = np.zeros(num_voxels)
        curvature_proxies = np.zeros(num_voxels)
        vertex_counts = np.zeros(num_voxels, dtype=int)
        
        avg_colors = []
        color_variances = []
        
        voxel_keys = []
        for i, (voxel_key, props) in enumerate(self.voxel_properties.items()):
            voxel_keys.append(voxel_key)
            centers[i] = self.voxel_centers[voxel_key]
            avg_normals[i] = props['average_normal']
            median_normals[i] = props['median_normal']
            normal_variances[i] = props['normal_variance']
            curvature_proxies[i] = props['curvature_proxy']
            vertex_counts[i] = props['num_vertices']
            
            if props['average_color'] is not None:
                avg_colors.append(props['average_color'])
                color_variances.append(props['color_variance'])
        
        result = {
            'centers': centers,
            'average_normals': avg_normals,
            'median_normals': median_normals,
            'normal_variances': normal_variances,
            'curvature_proxies': curvature_proxies,
            'vertex_counts': vertex_counts,
            'voxel_keys': voxel_keys
        }
        
        if avg_colors:
            result['average_colors'] = np.array(avg_colors)
            result['color_variances'] = np.array(color_variances)
        
        return result
    
    def visualize_in_rerun(self, session_name: str = "voxel_mesh",
                          show_normals: bool = True,
                          show_curvature: bool = True,
                          normal_scale: float = 0.02):
        """
        Visualize voxel properties in Rerun.
        
        Args:
            session_name: Name of the Rerun session
            show_normals: Whether to show average normal vectors
            show_curvature: Whether to color voxels by curvature
            normal_scale: Scale factor for normal vector visualization
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        rr.init(session_name, spawn=True)
        
        logger.info("Visualizing voxel mesh in Rerun...")
        
        # Get property arrays
        props = self.get_voxel_property_arrays()
        centers = props['centers']
        avg_normals = props['average_normals']
        curvature = props['curvature_proxies']
        
        # Visualize voxel centers as points
        if show_curvature:
            # Normalize curvature for color mapping (0 = blue/low, 1 = red/high)
            curvature_normalized = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
            colors = self._curvature_to_color(curvature_normalized)
            
            rr.log(
                "voxels/centers",
                rr.Points3D(
                    positions=centers,
                    colors=colors,
                    radii=self.voxel_size * 0.3
                )
            )
        elif 'average_colors' in props:
            rr.log(
                "voxels/centers",
                rr.Points3D(
                    positions=centers,
                    colors=props['average_colors'],
                    radii=self.voxel_size * 0.3
                )
            )
        else:
            rr.log(
                "voxels/centers",
                rr.Points3D(
                    positions=centers,
                    radii=self.voxel_size * 0.3
                )
            )
        
        # Visualize normals
        if show_normals:
            normal_length = self.mesh.extents.max() * normal_scale
            end_points = centers + avg_normals * normal_length
            
            # Log all normal vectors as line segments in a single static call
            line_strips = [np.array([start, end]) for start, end in zip(centers, end_points)]
            rr.log(
                "voxels/normals",
                rr.LineStrips3D(strips=line_strips)
            )
        
        logger.info("Voxel visualization logged to Rerun")
    
    def _curvature_to_color(self, curvature_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized curvature values to RGB colors (blue to red colormap).
        
        Args:
            curvature_normalized: Normalized curvature values (0-1)
            
        Returns:
            RGB colors array
        """
        colors = np.zeros((len(curvature_normalized), 3), dtype=np.uint8)
        
        # Blue (low curvature) to Red (high curvature) via green
        colors[:, 2] = (255 * (1 - curvature_normalized)).astype(np.uint8)  # Blue
        colors[:, 0] = (255 * curvature_normalized).astype(np.uint8)         # Red
        colors[:, 1] = (255 * (1 - np.abs(curvature_normalized - 0.5) * 2)).astype(np.uint8)  # Green
        
        return colors
    
    def get_high_curvature_voxels(self, threshold_percentile: float = 90) -> np.ndarray:
        """
        Get voxel centers with high curvature (potential edge/feature locations).
        
        Args:
            threshold_percentile: Percentile threshold for high curvature (0-100)
            
        Returns:
            Array of voxel centers with high curvature
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        props = self.get_voxel_property_arrays()
        curvature = props['curvature_proxies']
        centers = props['centers']
        
        threshold = np.percentile(curvature, threshold_percentile)
        high_curvature_mask = curvature >= threshold
        
        logger.info(f"Found {high_curvature_mask.sum()} high-curvature voxels "
                   f"(threshold: {threshold:.4f})")
        
        return centers[high_curvature_mask]
    
    def get_statistics(self) -> dict:
        """
        Get overall statistics about the voxel mesh.
        
        Returns:
            Dictionary containing voxel mesh statistics
        """
        if not self.voxel_properties:
            self.compute_voxel_properties()
        
        props = self.get_voxel_property_arrays()
        
        stats = {
            'num_voxels': len(self.voxel_properties),
            'voxel_size': self.voxel_size,
            'avg_vertices_per_voxel': np.mean(props['vertex_counts']),
            'normal_variance_mean': np.mean(props['normal_variances']),
            'normal_variance_std': np.std(props['normal_variances']),
            'curvature_proxy_mean': np.mean(props['curvature_proxies']),
            'curvature_proxy_std': np.std(props['curvature_proxies']),
            'curvature_proxy_max': np.max(props['curvature_proxies'])
        }
        
        if 'average_colors' in props:
            stats['color_variance_mean'] = np.mean(props['color_variances'])
            stats['color_variance_std'] = np.std(props['color_variances'])
        
        return stats
