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
    bucket: str,
    outfile: str
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
    IMAGE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/image_name -H "Metadata-Flavor: Google")
    CONTAINER_PARAM=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/container_param -H "Metadata-Flavor: Google")
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo gcloud auth configure-docker
    sudo docker pull {docker_image}
    sudo docker run -e BETA={beta} -e EPSILON={epsilon} -e SIMULATIONS={simulations} -e DAYS={days} -e GCS_BUCKET={bucket} -e OUTFILENAME={outfile} {docker_image}""" + """
    zoneMetadata=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor:Google")
    IFS=$'/'
    zoneMetadataSplit=($zoneMetadata)
    ZONE="${zoneMetadataSplit[3]}"
    docker run --entrypoint "gcloud" google/cloud-sdk:alpine compute instances delete ${HOSTNAME}  --delete-disks=all --zone=${ZONE}
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


from google.cloud import compute_v1
from google.oauth2 import service_account

# Load credentials for the InstancesClient
credentials = service_account.Credentials.from_service_account_file(
    "C:/Users/00011/Downloads/quiet-fusion-440218-i2-58c9e4633660.json"
)
instances_client = compute_v1.InstancesClient(credentials=credentials)
def create_instance_with_image(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    source_image: str,
    beta: float,
    epsilon: float,
    simulations: int,
    days: int,
    bucket: str,
    outfile: str
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance with full API access that pulls and runs a Docker container.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        source_image: Full path to the source image, e.g., "projects/my-project/global/images/my-custom-image".
        docker_image: Docker image to pull and run.
        beta: Value for the beta environment variable.
        epsilon: Value for the epsilon environment variable.
        simulations: Number of simulations to run.
        days: Number of days to simulate.
        bucket: GCS bucket to use.
        outfile: Output file name.

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
    cd ..
    cd ..
    cd opt/myapp
    source venv/bin/activate
    export BETA={beta} EPSILON={epsilon} SIMULATIONS={simulations} DAYS={days} GCS_BUCKET={bucket} OUTFILENAME={outfile}
    python main.py""" + """
    gcloud compute instances delete ${HOSTNAME} --delete-disks=all --zone=$(curl -H "Metadata-Flavor:Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F'/' '{print $4}') --quiet
    """


    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    # Define the disk for the VM using the specified custom image
    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=source_image,
        disk_size_gb=10,  # (we can make this a param in the future)
        disk_type=f"zones/{zone}/diskTypes/pd-balanced"
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )
    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config]
    )

    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        ]
    )
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()

    return instances_client.get(project=project_id, zone=zone, instance=instance_name)




def create_instance_and_save_image(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    image_family: str,
    image_project: str,
    bucket_name: str,
    script_name: str,
    requirements_name: str,
    custom_image_name: str
) -> compute_v1.Instance:
    """
    Creates a Compute Engine VM instance, sets up a persistent venv, downloads a script and requirements,
    verifies the setup using logs, and saves the instance as a custom image.

    Args:
        project_id: ID or number of the project you want to use.
        zone: Name of the zone you want to check, for example: us-west3-b.
        instance_name: Name of the new instance.
        machine_type: Machine type for the VM, e.g., "e2-medium".
        image_family: Image family for the VM OS, e.g., "debian-12".
        image_project: Google Cloud project hosting the OS image.
        bucket_name: GCP Storage bucket name containing the script and requirements.
        script_name: Name of the script to download from the bucket.
        requirements_name: Name of the requirements file to download from the bucket.
        custom_image_name: Name of the custom image to create.

    Returns:
        Instance object.
    """
    startup_script = f"""#!/bin/bash
sudo apt-get update
sudo apt-get install python3-pip gsutil

mkdir -p /opt/myapp/
chmod -R 755 /opt/myapp/

gsutil cp gs://{bucket_name}/{script_name} /opt/myapp/main.py
gsutil cp gs://{bucket_name}/{requirements_name} /opt/myapp/requirements.txt

cd ..
cd ..
cd opt/myapp
sudo apt install -y python3-pip
sudo apt install -y python3.11-venv
sudo python3 -m venv /opt/myapp/venv
source venv/bin/activate
sudo /opt/myapp/venv/bin/pip install -r requirements.txt

"""

    metadata_items = [
        compute_v1.Items(key="startup-script", value=startup_script)
    ]

    initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=f"projects/{image_project}/global/images/family/{image_family}",
        disk_size_gb=10,
        disk_type=f"zones/{zone}/diskTypes/pd-balanced"
    )
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=initialize_params
    )

    access_config = compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")
    network_interface = compute_v1.NetworkInterface(
        name="global/networks/default",
        access_configs=[access_config]
    )

    instance_resource = compute_v1.Instance(
        name=instance_name,
        machine_type=f"zones/{zone}/machineTypes/{machine_type}",
        disks=[disk],
        network_interfaces=[network_interface],
        metadata=compute_v1.Metadata(items=metadata_items),
        service_accounts=[
            compute_v1.ServiceAccount(
                email="default",
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        ]
    )

    instances_client = compute_v1.InstancesClient()
    instance_insert_request = compute_v1.InsertInstanceRequest(
        project=project_id,
        zone=zone,
        instance_resource=instance_resource
    )
    operation = instances_client.insert(request=instance_insert_request)
    operation.result()  # Wait for the operation to complete

    instances_client = compute_v1.InstancesClient()
    stop_request = compute_v1.StopInstanceRequest(
        project=project_id,
        zone=zone,
        instance=instance_name
    )
    stop_operation = instances_client.stop(request=stop_request)
    stop_operation.result()  # Wait for the instance to stop

    # Create the custom image
    images_client = compute_v1.ImagesClient()
    image_request = compute_v1.InsertImageRequest(
        project=project_id,
        image_resource=compute_v1.Image(
            name=custom_image_name,
            source_disk=f"projects/{project_id}/zones/{zone}/disks/{instance_name}"
        )
    )
    image_operation = images_client.insert(request=image_request)
    image_operation.result()  # Wait for the image creation to complete

    delete_request = compute_v1.DeleteInstanceRequest(
        project = project_id,
        instance=instance_name,
        zone=zone
    )

    instances_client.delete(request=delete_request)
    stop_operation.result()

    return custom_image_name

