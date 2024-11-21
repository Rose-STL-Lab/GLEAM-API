print ("test")

from google.cloud import compute_v1

print ("compute downloaded")

instances_client = compute_v1.InstancesClient()

print ("Instance client found")

def create_instance_with_docker(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    image_family: str,
    image_project: str,
    docker_image: str,
    beta: float,
    epsilon: float,
    simulations: int,
    days: int,
    bucket: str
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance with full API access that pulls and runs a Docker container.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        image_family: Image family for the VM OS, e.g., "debian-12".
        image_project: Google Cloud project hosting the OS image.
        docker_image: Docker container image to pull and run.
        beta: Value for the beta environment variable.
        epsilon: Value for the epsilon environment variable.

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo gcloud auth configure-docker
    sudo docker pull {docker_image}
    sudo docker run -e BETA={beta} -e EPSILON={epsilon} -e SIMULATIONS={simulations} -e DAYS={days} -e GCS_BUCKET={bucket} {docker_image}
    """

    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Define the disk for the VM using Debian 12 Bookworm First One I Found that worked, Could be changed for different workloads
    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image="projects/debian-cloud/global/images/debian-12-bookworm-v20241112",
        disk_size_gb=10,  # Specify 10GB disk size
        disk_type=f"zones/{zone}/diskTypes/pd-balanced" 
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )

    # Define the network interface with an external IP Important for Downloading 
    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config] 
    )

    # Define the instance
    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"] # Important for Pulling Docker Image
            )
        ]
    )

    # Insert the instance
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()  # Wait for the operation to complete

    return instances_client.get(project=project_id, zone=zone, instance=instance_name)
